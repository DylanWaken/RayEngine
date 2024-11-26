/**
 * @file rayEngine.h
 * @author Dylan Sun 
 * @brief Bounding Volume Hierarchy for ray tracing
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef BVH_H
#define BVH_H

#pragma once

#include "rayTypes.h"
#include "intersectPrimitives.h"
#include "memory.h"
#include "parallelFor.h"
#include <vector>
#include <iostream>
#define AABB_INVALID_INDEX 0xffffffff

namespace ray {
    template <typename T>
    struct BVHNode {
        AABB<T> aabb;
        uint32_t child1 = AABB_INVALID_INDEX;  // position of the first child in the BVH array
        uint32_t child2 = AABB_INVALID_INDEX;  // position of the second child in the BVH array
        uint32_t primitiveID = AABB_INVALID_INDEX;  // position of the primitive in the primitives array

        BVHNode() : aabb(), child1(AABB_INVALID_INDEX), child2(AABB_INVALID_INDEX), primitiveID(AABB_INVALID_INDEX) {}
    };

    /**
     * @brief Calculate the surface area of an AABB
     * @param aabb The AABB to calculate the surface area of
     * @return The surface area of the AABB
     */
    template <typename T>
    inline T calculateSurfaceArea(const AABB<T>& aabb) {
        Eigen::Vector<T, 3> extent = aabb.max - aabb.min;
        return 2.0 * (extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0]);
    }

    /**
     * @brief Calculate the volume of an AABB
     * @param aabb The AABB to calculate the volume of
     * @return The volume of the AABB
     */
    template <typename T>
    inline T calculateVolume(const AABB<T>& aabb) {
        Eigen::Vector<T, 3> extent = aabb.max - aabb.min;
        return extent[0] * extent[1] * extent[2];
    }

    /**
     * @brief Compute the bounds of a range of primitives
     * @param primitives The array of primitives
     * @param start The start index
     * @param end The end index
     * @return The bounds of the primitives
     */
    template <typename T>
    inline AABB<T> computeBounds(const std::vector<Primitive<T>*>& primitives, std::vector<uint32_t>& primitiveIndices, uint32_t start, uint32_t end) {
        Eigen::Vector<T, 3> min = primitives[primitiveIndices[start]]->getAABB().min;
        Eigen::Vector<T, 3> max = primitives[primitiveIndices[start]]->getAABB().max;
        
        for (uint32_t i = start + 1; i < end; i++) {
            const AABB<T>& primBounds = primitives[primitiveIndices[i]]->getAABB();
            min = min.cwiseMin(primBounds.min);
            max = max.cwiseMax(primBounds.max);
        }
        return AABB<T>(min, max);
    }

    /**
     * @brief Sort primitives along a given axis
     * @param primitives The array of primitives
     * @param start The start index
     * @param end The end index
     * @param axis The axis to sort along
     */
    template <typename T>
    inline void sortPrimitives(const std::vector<Primitive<T>*>& primitives, std::vector<uint32_t>& primitiveIndices, uint32_t start, uint32_t end, uint32_t axis) {
        std::sort(primitiveIndices.begin() + start, primitiveIndices.begin() + end,
            [&](uint32_t a, uint32_t b) {
                return primitives[a]->getAABB().center(axis) < primitives[b]->getAABB().center(axis);
            });
    }

    /**
     * @brief Build a BVH recursively using SAH (Surface Area Heuristic)
     * 
     * This algorithm is an idea that we BOUND the most amount of primitives in the SMALLEST possible sub-volume
     * 
     * @param start The start index
     * @param end The end index
     * @param nextNodeIndex The next available node index
     * @return The index of the current node
     */
    template <typename T>
    inline uint32_t buildRecursive(const std::vector<Primitive<T>*>& primitives, std::vector<BVHNode<T>>& nodes, std::vector<uint32_t>& primitiveIndices, 
        uint32_t start, uint32_t end, uint32_t& nextNodeIndex, T traversalCost = 1.0, T intersectionCost = 2.0) {
        
        // we layout nodes in an DFS order
        uint32_t nodeIndex = nextNodeIndex;
        nextNodeIndex++;
        nodes.emplace_back();
        
        // Create leaf node if we have few enough primitives
        if (end - start <= 1) {
            nodes[nodeIndex].primitiveID = primitiveIndices[start];
            nodes[nodeIndex].aabb = primitives[primitiveIndices[start]]->getAABB();
            return nodeIndex;
        }

        // Compute bounds for current node
        nodes[nodeIndex].aabb = computeBounds(primitives, primitiveIndices, start, end);
        
        // Find best split using SAH
        // The Cost is the total surface area produced by the split (i.e. the left and right bounds)
        T bestCost = std::numeric_limits<T>::infinity();
        uint32_t bestAxis = 0;
        uint32_t bestSplit = start;
        
        // Try each axis (x, y, z)
        for (uint32_t axis = 0; axis < 3; axis++) {
            // Sort primitives along current axis
            sortPrimitives(primitives, primitiveIndices, start, end, axis);
                        
            // Try different split positions
            AABB<T> leftBounds = primitives[primitiveIndices[start]]->getAABB();
            std::vector<AABB<T>> rightBounds;
            rightBounds.reserve(end - start);

            // initialize right bounds
            for (uint32_t i = 0; i < end - start; i++) {
                rightBounds.push_back(primitives[primitiveIndices[start+i]]->getAABB());
            }
            
            // Precompute right bounds
            for (int32_t i = end-start-2; i >= 0; i--) {
                // we keep adding more primitives to the right bound as we iterate to create bounds that covers incrementally the right side
                rightBounds[i].min = rightBounds[i].min.cwiseMin(rightBounds[i+1].min);
                rightBounds[i].max = rightBounds[i].max.cwiseMax(rightBounds[i+1].max);
            }
            
            // Evaluate splits
            for (uint32_t mid = start + 1; mid < end; mid++) {
                leftBounds.min = leftBounds.min.cwiseMin(
                    primitives[primitiveIndices[mid-1]]->getAABB().min);
                leftBounds.max = leftBounds.max.cwiseMax(
                    primitives[primitiveIndices[mid-1]]->getAABB().max);
                
                T leftArea = calculateSurfaceArea(leftBounds);
                T rightArea = calculateSurfaceArea(rightBounds[mid-start]);
                
                // C = Traversal + Intersection * (Left Area * Left Count + Right Area * Right Count) / Total Area
                T cost = traversalCost + intersectionCost * 
                    (leftArea * (mid - start) + rightArea * (end - mid)) / 
                    calculateSurfaceArea(nodes[nodeIndex].aabb);
                
                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = mid;
                }
            }
        }
        
        // Sort along best axis
        sortPrimitives(primitives, primitiveIndices, start, end, bestAxis);
            
        // Create child nodes
        // this will cover the entire left sub-tree
        nodes[nodeIndex].child1 = buildRecursive(primitives, nodes, primitiveIndices, start, bestSplit,
            nextNodeIndex, traversalCost, intersectionCost);
        // this will cover the entire right sub-tree
        nodes[nodeIndex].child2 = buildRecursive(primitives, nodes, primitiveIndices, bestSplit, end, 
            nextNodeIndex, traversalCost, intersectionCost);
        
        return nodeIndex;
    }

    /**
     * @brief Build a standard binary BVH from an array of primitives (no optimizations)
     * @param primitives The array of primitives
     * @param count The number of primitives
     * @param nodes The array of BVH nodes
     * @param nodeCount The number of nodes
     */
    template <typename T>
    void buildBinaryBVH(const std::vector<Primitive<T>*>& primitives, std::vector<BVHNode<T>>& nodes) {
        std::cout << "<RayEngine> Building BVH..." << std::endl;
        uint32_t count = primitives.size();
        if (count <= 0) {
            throw std::invalid_argument("Count must be greater than 0");
        }

        std::vector<uint32_t> primitiveIndices(count);

        for (int32_t i = 0; i < count; i++) {
            primitiveIndices[i] = i;
        }

        // Start recursive build
        uint32_t nextNodeIndex = 0;
        buildRecursive(primitives, nodes, primitiveIndices, 0, count, nextNodeIndex);
        std::cout << "<RayEngine> BVH built with " << nextNodeIndex << "nodes." << std::endl;
    }
}

#endif  
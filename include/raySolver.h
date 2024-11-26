/**
 * @file rayEngine.h
 * @author Dylan Sun 
 * @brief Ray solver for ray tracing, recursive intersection testing in parallel
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RAY_SOLVER_H
#define RAY_SOLVER_H

#pragma once

#include "rayTypes.h"
#include "intersectPrimitives.h"
#include "bvh.h"

namespace ray{

    /**
     * @brief Trace a ray through the scene
     * @param ray The ray to trace
     * @param primitives The primitives in the scene
     * @param bvhNodes The BVH nodes in the scene
     * @return The intersection result
     */
    template <typename T>
    inline Intersection<T> trace(const Ray<T>& ray, const std::vector<Primitive<T>*>& primitives, const std::vector<BVHNode<T>>& bvhNodes){
        // Intersection result
        Intersection<T> intersection;
        intersection.hit = false;

        // Create a stack to store BVH nodes for traversal
        std::vector<uint32_t> nodeStack;
        nodeStack.reserve(64);
        nodeStack.push_back(0);

        // find the first intersection
        T tMin = std::numeric_limits<T>::infinity();

        // Iterate through the BVH nodes
        while (!nodeStack.empty()){
            uint32_t nodeIndex = nodeStack.back();
            nodeStack.pop_back();

            // Check if the current node is a leaf node
            const BVHNode<T>& node = bvhNodes[nodeIndex];

            // Skip if the ray does not intersect with the current node's AABB
            if (!node.aabb.intersect(ray)) continue;

            // Check if the current node is a leaf node, if so, test intersection
            if (node.primitiveID != AABB_INVALID_INDEX){
                const Primitive<T>* primitive = primitives[node.primitiveID];
                Intersection<T> currentIntersection = primitive->intersect(ray);

                // Update the intersection result if the current intersection is closer
                if (currentIntersection.hit && currentIntersection.tHit < tMin){
                    tMin = currentIntersection.tHit;
                    intersection = currentIntersection;
                }
            } else {
                // Push the children of the current node onto the stack
                if (node.child1 != AABB_INVALID_INDEX) nodeStack.push_back(node.child1);
                if (node.child2 != AABB_INVALID_INDEX) nodeStack.push_back(node.child2);
            }
        }

        return intersection;
    }

    /**
     * @brief Check if two points are visible to each other (used for final direct lighting)
     * @param point1 The first point
     * @param point2 The second point
     * @param primitives The primitives in the scene
     * @param bvhNodes The BVH nodes in the scene
     * @return True if the points are visible to each other, false otherwise
     */
    template <typename T>
    inline bool visible(const Eigen::Vector3<T>& point1, const Eigen::Vector3<T>& point2, const std::vector<Primitive<T>*>& primitives, const std::vector<BVHNode<T>>& bvhNodes, uint32_t leavingPrimitiveID){
        // Calculate direction without normalizing
        Eigen::Vector3<T> direction = point2 - point1;
        T distanceToP2 = direction.norm();
        
        // Create ray with normalized direction (required for intersection testing)
        Ray<T> ray(point1, direction.normalized(), leavingPrimitiveID);
        Intersection<T> intersection = trace(ray, primitives, bvhNodes);
        
        // Add a small epsilon to handle numerical precision issues
        const T epsilon = static_cast<T>(1e-4);
        bool visible =  !intersection.hit || intersection.tHit > distanceToP2 - epsilon;

        return visible;
    }
}

#endif
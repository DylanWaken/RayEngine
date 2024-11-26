/**
 * @file rayTypes.h
 * @author Dylan Sun 
 * @brief Types for ray tracing
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RAY_TYPES_H
#define RAY_TYPES_H

#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <limits>

#define PRIMITIVE_INVALID_INDEX 0xFFFFFFFF

namespace ray{
    /**
     * @brief Ray structure
     */
    template <typename T>
    struct Ray {
        // origin of the ray
        Eigen::Vector<T, 3> origin;
        // normalized direction vector
        Eigen::Vector<T, 3> direction;
        // leaving primitive ID
        uint32_t leavingPrimitiveID;

        Ray(const Eigen::Vector<T, 3>& origin, const Eigen::Vector<T, 3>& direction, uint32_t leavingPrimitiveID) : origin(origin), direction(direction), leavingPrimitiveID(leavingPrimitiveID) {}
        Ray(const Eigen::Vector<T, 3>& origin, const Eigen::Vector<T, 3>& direction) : origin(origin), direction(direction), leavingPrimitiveID(PRIMITIVE_INVALID_INDEX) {}
        Ray() : origin(Eigen::Vector<T, 3>::Zero()), direction(Eigen::Vector<T, 3>::Zero()), leavingPrimitiveID(PRIMITIVE_INVALID_INDEX) {}

        inline Eigen::Vector<T, 3> operator()(T t) const {
            return origin + t * direction;
        }
    };

    /**
     * @brief Intersection structure
     */
    template <typename T>
    struct Intersection{
        Eigen::Vector<T, 3> position;
        Eigen::Vector<T, 3> normal;
        Eigen::Vector<T, 3> tangent;
        Eigen::Vector<T, 2> uv;
        T tHit;
        uint32_t primitiveID;
        bool hit = false;

        Intersection() : 
            position(Eigen::Vector<T, 3>::Zero()), 
            normal(Eigen::Vector<T, 3>::Zero()), 
            tangent(Eigen::Vector<T, 3>::Zero()),
            uv(Eigen::Vector<T, 2>::Zero()),
            tHit(std::numeric_limits<T>::infinity()), 
            primitiveID(0), 
            hit(false) {}

        // Intersection with uv coordinates 
        Intersection(const Eigen::Vector<T, 3>& position, const Eigen::Vector<T, 3>& normal, const Eigen::Vector<T, 3>& tangent, const Eigen::Vector<T, 2>& uv, T tHit, uint32_t primitiveID, bool hit) :
            position(position), normal(normal), tangent(tangent), 
            uv(uv), tHit(tHit), primitiveID(primitiveID), hit(hit) {}
    };
}

#endif

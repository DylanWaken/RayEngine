/**
 * @file rayEngine.h
 * @author Dylan Sun 
 * @brief Primitive shapes for ray tracing
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include "utils.h"
#include "principleBRDF.h"
#include "rayTypes.h"   
#include <iostream>

#define SPHERE_PRIMITIVE 0
#define TRIANGLE_PRIMITIVE 1
#define MIN_T_HIT 1e-3

namespace ray{
    /**
     * @brief Axis-aligned bounding box structure
     */
    template <typename T>
    struct AABB {
        Eigen::Vector<T, 3> min;
        Eigen::Vector<T, 3> max;

        AABB() : min(Eigen::Vector<T, 3>::Zero()), max(Eigen::Vector<T, 3>::Zero()) {}
        AABB(const Eigen::Vector<T, 3>& min, const Eigen::Vector<T, 3>& max) : min(min), max(max) {}

        /**
         * @brief Intersect the AABB with a ray
         * 
         * AABB intersection works by finding the intersection distances (t) along each axis
         * where the ray enters and exits the box boundaries.
         * 
         * @param ray The ray to intersect with
         * @return True if the ray intersects the AABB, false otherwise
         */
        inline bool intersect(const Ray<T>& ray) const{
            // Compute inverse direction (we avoid more divisions by doing this)
            Eigen::Vector<T, 3> dirfrac;
            dirfrac[0] = 1.0 / ray.direction[0];
            dirfrac[1] = 1.0 / ray.direction[1]; 
            dirfrac[2] = 1.0 / ray.direction[2];

            // Calculate intersection distances for each pair of planes
            T t1 = (min[0] - ray.origin[0]) * dirfrac[0];
            T t2 = (max[0] - ray.origin[0]) * dirfrac[0];
            T t3 = (min[1] - ray.origin[1]) * dirfrac[1];
            T t4 = (max[1] - ray.origin[1]) * dirfrac[1];
            T t5 = (min[2] - ray.origin[2]) * dirfrac[2];
            T t6 = (max[2] - ray.origin[2]) * dirfrac[2];

            // Find largest minimum and smallest maximum
            T tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
            T tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

            // Check if intersection is behind ray origin
            if (tmax < 0)
            {
                return false;
            }

            // Check if ray misses box
            if (tmin > tmax)
            {
                return false;
            }

            return true;
        }

        /**
         * @brief Get the center of the AABB
         * @return The center of the AABB
         */
        inline Eigen::Vector<T, 3> center() const {
            return (min + max) * 0.5;
        }

        inline T center(uint32_t axisID) const {
            return (min[axisID] + max[axisID]) * 0.5;
        }

        /**
         * @brief Project and rasterize the AABB, for debug rasterization
         */
        inline void projectAndRasterize(
            const Eigen::Matrix<T, 4, 4>& projectionMatrix, 
            const Eigen::Matrix<T, 4, 4>& world2cam, 
            uint8_t* frameBuffer, 
            uint32_t width, 
            uint32_t height
        ) const {
            // Get the 8 vertices of the AABB in local space
            Eigen::Vector<T, 4> vertices[8] = {
                {min[0], min[1], min[2], 1.0},
                {max[0], min[1], min[2], 1.0},
                {min[0], max[1], min[2], 1.0}, 
                {max[0], max[1], min[2], 1.0},
                {min[0], min[1], max[2], 1.0},
                {max[0], min[1], max[2], 1.0},
                {min[0], max[1], max[2], 1.0},
                {max[0], max[1], max[2], 1.0}
            };

            // Transform all vertices to screen space
            Eigen::Vector<T, 2> screen_vertices[8];
            for(int i = 0; i < 8; i++) {
                // Transform to clip space
                Eigen::Vector<T, 4> clip = projectionMatrix * world2cam * vertices[i];
                
                // Perspective divide
                clip /= clip[3];
                
                // Convert to screen coordinates
                screen_vertices[i] = Eigen::Vector<T, 2>(
                    (clip[0] + 1) * width * 0.5,
                    (clip[1] + 1) * height * 0.5
                );

            }

            const Eigen::Vector<uint8_t, 3> color = Eigen::Vector<uint8_t, 3>{255,0,0};
            // Draw all edges of the cube
            // Bottom face
            drawLine(screen_vertices[0], screen_vertices[1], frameBuffer, width, height, color);
            drawLine(screen_vertices[1], screen_vertices[3], frameBuffer, width, height, color);
            drawLine(screen_vertices[3], screen_vertices[2], frameBuffer, width, height, color);
            drawLine(screen_vertices[2], screen_vertices[0], frameBuffer, width, height, color);

            // Top face
            drawLine(screen_vertices[4], screen_vertices[5], frameBuffer, width, height, color);
            drawLine(screen_vertices[5], screen_vertices[7], frameBuffer, width, height, color);
            drawLine(screen_vertices[7], screen_vertices[6], frameBuffer, width, height, color);
            drawLine(screen_vertices[6], screen_vertices[4], frameBuffer, width, height, color);

            // Connecting edges
            drawLine(screen_vertices[0], screen_vertices[4], frameBuffer, width, height, color);
            drawLine(screen_vertices[1], screen_vertices[5], frameBuffer, width, height, color);
            drawLine(screen_vertices[2], screen_vertices[6], frameBuffer, width, height, color);
            drawLine(screen_vertices[3], screen_vertices[7], frameBuffer, width, height, color);
        }
    };

    /**
     * @brief Base class for all primitives
     * @tparam T The data precision of the primitive, usually double or float
     * 
     * TODO: Figure out a better way for decreasing the overhead of virtual functions
     */
    template <typename T>
    struct Primitive {
        /**
         * @brief Intersect the primitive with a ray
         * @param ray The ray to intersect with
         * @return The intersection information
         */
        virtual Intersection<T> intersect(const Ray<T>& ray) const = 0;

        /**
         * @brief Get the bounding box of the primitive
         * @return The bounding box of the primitive
         */
        virtual AABB<T> getAABB() const = 0;

        /**
         * @brief Project and rasterize the primitive, for debug rasterization
         * @param projectionMatrix The projection matrix
         * @param world2cam The world to camera matrix
         * @param frameBuffer The frame buffer (CHW)
         * @param width The width of the frame buffer
         * @param height The height of the frame buffer
         */
        virtual void projectAndRasterize(
            const Eigen::Matrix<T, 4, 4>& projectionMatrix, 
            const Eigen::Matrix<T, 4, 4>& world2cam, 
            uint8_t* frameBuffer, 
            uint32_t width, 
            uint32_t height
        ) const = 0;

        /**
         * @brief The type of the primitive
         */
        virtual uint32_t type() const = 0;

        /**
         * @brief The BRDF of the primitive (spatial constant)
         */
        BRDF<T>* brdf;
        uint32_t primitiveID;
    };

    /**
     * @brief Sphere primitive
     * @tparam T The data precision of the primitive, usually double or float
     */
    template <typename T>
    struct Sphere : Primitive<T> {
        Eigen::Vector<T, 3> center;
        T radius;

        Sphere(const Eigen::Vector<T, 3>& center, T radius) : center(center), radius(radius) {}

        /**
         * @brief Intersect the sphere with a ray
         * Pythagorean theorem.
         */
        Intersection<T> intersect(const Ray<T>& ray) const override {
            // if the ray is leaving this primitive, return no intersection
            if (ray.leavingPrimitiveID == this->primitiveID){
                return Intersection<T>();
            }
            // Get vector from ray origin to sphere center (m = p - s.c) 
            Eigen::Vector<T, 3> m = ray.origin - center;
            
            // Compute b = Dot(m, d) |m| * cos(theta) : the projection length of the edge from the ray origin to the sphere center onto the ray direction
            // c = |m|^2 - r^2 
            T b = m.dot(ray.direction);
            T c = m.dot(m) - radius * radius;
            
            // Exit if ray origin inside sphere (c <= 0) 
            if (c <= 0) {
                return Intersection<T>();
            }
            
            // now we do an equation substitution to solve for t, since m = o - center
            // |m + t * d|^2 - r^2 = 0 for desired t
            // substitute t^2 * |d|^2 + 2 * t * dot(d, m) + |m|^2 - r^2 = 0
            // by letting b = dot(m,d), c = |m|^2 - r^2, and |d| = 1
            // t^2 + 2tb + c = 0
            // and we solve this like a quadratic equation to get t

            // Solve quadratic equation t^2 + (2b)t + c = 0
            T discriminant = b * b - c;
            
            // No real roots means no intersection
            if (discriminant <= 0) {
                return Intersection<T>();
            }
            
            // Get smallest positive root -b - sqrt(discriminant)
            T tHit = -b - std::sqrt(discriminant);

            // Exit if the ray hits the sphere behind the origin
            if (tHit < MIN_T_HIT) return Intersection<T>();
            
            // Calculate hit point
            Eigen::Vector<T, 3> hitPoint = ray.origin + tHit * ray.direction;
            // Calculate surface normal at hit point
            Eigen::Vector<T, 3> normal = (hitPoint - center).normalized();
            // uv as latitude and longitude 
            Eigen::Vector<T, 2> uv = Eigen::Vector<T, 2>(
                std::atan2(normal[0], normal[2]) / (2 * M_PI),
                std::asin(normal[1]) / M_PI
            );

            // tangent is longitude direction
            Eigen::Vector<T, 3> tangent = Eigen::Vector<T, 3>(
                -normal[1],
                normal[0],
                0
            ).normalized();

            return Intersection<T>(hitPoint, normal, tangent, uv, tHit, this->primitiveID, true);
        }

        /**
         * @brief Get the bounding box of the sphere
         */
        AABB<T> getAABB() const override {
            // The AABB extends radius units in each direction from the center
            Eigen::Vector<T, 3> min = center - Eigen::Vector<T, 3>::Constant(radius);
            Eigen::Vector<T, 3> max = center + Eigen::Vector<T, 3>::Constant(radius);
            return AABB<T>(min, max);
        }

        uint32_t type() const override {
            return SPHERE_PRIMITIVE;
        }

        void projectAndRasterize(
            const Eigen::Matrix<T, 4, 4>& projectionMatrix, 
            const Eigen::Matrix<T, 4, 4>& world2cam, 
            uint8_t* frameBuffer, 
            uint32_t width, 
            uint32_t height
        ) const override {

        }
    };

    /**
     * @brief Triangle primitive
     * @tparam T The data precision of the primitive, usually double or float
     */
    template <typename T>
    struct Triangle : Primitive<T> {
        Eigen::Vector<T, 3> v0, v1, v2;
        Eigen::Vector<T, 3> n0, n1, n2; // per-vertex normal vectors (used to compute local geometry normal)
        Eigen::Vector<T, 2> uv0, uv1, uv2; // per-vertex uv coordinates
        Eigen::Vector<T, 3> normal; // normal for the primitive (used for primitive intersection)
        Eigen::Vector<T, 3> tangent; // tangent for the primitive (used for principle BRDF) in the forward U direction of uv space

        Triangle(const Eigen::Vector<T, 3>& v0, 
                 const Eigen::Vector<T, 3>& v1, 
                 const Eigen::Vector<T, 3>& v2) : 
                 
                 v0(v0), 
                 v1(v1), 
                 v2(v2), 

                 // default uv coordinates for vertex maps
                 uv0({0,0}), 
                 uv1({0,1}), 
                 uv2({1,0}) {
                    normal = Eigen::Vector<T, 3>((v1 - v0).cross(v2 - v0)).normalized();
                    n0 = n1 = n2 = normal;
                    tangent = findUForward(v0, v1, v2, uv0, uv1, uv2);
                 }

        Triangle(const Eigen::Vector<T, 3>& v0, 
                 const Eigen::Vector<T, 3>& v1, 
                 const Eigen::Vector<T, 3>& v2, 
                 const Eigen::Vector<T, 3>& n0, 
                 const Eigen::Vector<T, 3>& n1, 
                 const Eigen::Vector<T, 3>& n2) : 
                 v0(v0), 
                 v1(v1), 
                 v2(v2), 
                 n0(n0), 
                 n1(n1), 
                 n2(n2),
                 uv0({0,0}), 
                 uv1({0,1}), 
                 uv2({1,0}) {
                    normal = Eigen::Vector<T, 3>((v1 - v0).cross(v2 - v0)).normalized();
                    tangent = findUForward(v0, v1, v2, uv0, uv1, uv2);
                 }

        Triangle(const Eigen::Vector<T, 3>& v0, 
                 const Eigen::Vector<T, 3>& v1, 
                 const Eigen::Vector<T, 3>& v2, 
                 const Eigen::Vector<T, 3>& n0, 
                 const Eigen::Vector<T, 3>& n1, 
                 const Eigen::Vector<T, 3>& n2, 
                 const Eigen::Vector<T, 2>& uv0, 
                 const Eigen::Vector<T, 2>& uv1, 
                 const Eigen::Vector<T, 2>& uv2) : 
                 v0(v0), 
                 v1(v1), 
                 v2(v2), 
                 n0(n0), 
                 n1(n1), 
                 n2(n2), 
                 uv0(uv0), 
                 uv1(uv1), 
                 uv2(uv2) {
                    normal = Eigen::Vector<T, 3>((v1 - v0).cross(v2 - v0)).normalized();
                    tangent = findUForward(v0, v1, v2, uv0, uv1, uv2);
                 }

        Triangle(const Eigen::Vector<T, 3>& v0, 
                 const Eigen::Vector<T, 3>& v1, 
                 const Eigen::Vector<T, 3>& v2, 
                 const Eigen::Vector<T, 2>& uv0, 
                 const Eigen::Vector<T, 2>& uv1, 
                 const Eigen::Vector<T, 2>& uv2) : 
                 v0(v0), 
                 v1(v1), 
                 v2(v2), 
                 uv0(uv0), 
                 uv1(uv1), 
                 uv2(uv2) {
                    normal = Eigen::Vector<T, 3>((v1 - v0).cross(v2 - v0)).normalized();
                    n0 = n1 = n2 = normal;
                    tangent = findUForward(v0, v1, v2, uv0, uv1, uv2);
                 }

        /**
         * @brief Intersect the triangle with a ray
         */
        Intersection<T> intersect(const Ray<T>& ray) const override {
            // if the ray is leaving this primitive, return no intersection
            if (ray.leavingPrimitiveID == this->primitiveID){
                return Intersection<T>();
            }
            // Compute denominator for intersection test
            T denom = normal.dot(ray.direction);

            // Check if ray is parallel to triangle plane
            if (std::abs(denom) < std::numeric_limits<T>::epsilon()) {
                return Intersection<T>();
            }

            // Compute distance from ray origin to triangle plane
            T d = normal.dot(v0);
            T t_hit = (d - normal.dot(ray.origin)) / denom;

            // Check if intersection is behind ray origin
            if (t_hit < MIN_T_HIT) {
                return Intersection<T>();
            }

            // Compute intersection point
            Eigen::Vector<T, 3> hit_point = ray.origin + t_hit * ray.direction;

            // Inside-outside test using barycentric coordinates
            Eigen::Vector<T, 3> edge0 = v1 - v0;
            Eigen::Vector<T, 3> edge1 = v2 - v1;
            Eigen::Vector<T, 3> edge2 = v0 - v2;
            Eigen::Vector<T, 3> c0 = hit_point - v0;
            Eigen::Vector<T, 3> c1 = hit_point - v1;
            Eigen::Vector<T, 3> c2 = hit_point - v2;

            // Check if point is inside all edge planes
            if (normal.dot(edge0.cross(c0)) < 0 ||
                normal.dot(edge1.cross(c1)) < 0 ||
                normal.dot(edge2.cross(c2)) < 0) {
                return Intersection<T>();
            }

            // Compute barycentric coordinates for normal interpolation
            T area = normal.dot(edge0.cross(edge2)); // Total triangle area * 2
            T u = normal.dot(edge0.cross(c0)) / area; // Barycentric coordinate for v2
            T v = normal.dot(edge1.cross(c1)) / area; // Barycentric coordinate for v0
            T w = normal.dot(edge2.cross(c2)) / area; // Barycentric coordinate for v1

            // The uv coordinates are interpolated using barycentric coordinates
            Eigen::Vector<T, 2> uv = (v * uv0 + w * uv1 + u * uv2);
            
            Eigen::Vector<T, 3> normalOut = (v * n0 + w * n1 + u * n2).normalized();

            return Intersection<T>(hit_point, normalOut, tangent, uv, t_hit, this->primitiveID, true);
        }

        /**
         * @brief Get the bounding box of the triangle
         */
        AABB<T> getAABB() const override {
            // Initialize min and max vectors with the first vertex
            Eigen::Vector<T, 3> min_point = v0;
            Eigen::Vector<T, 3> max_point = v0;

            // Compare with v1 and update min/max
            min_point = min_point.cwiseMin(v1);
            max_point = max_point.cwiseMax(v1);

            // Compare with v2 and update min/max
            min_point = min_point.cwiseMin(v2);
            max_point = max_point.cwiseMax(v2);

            return AABB<T>(min_point, max_point);
        }

        uint32_t type() const override {
            return TRIANGLE_PRIMITIVE;
        }

        /**
         * @brief Project and rasterize the triangle, for debug rasterization
         */
        void projectAndRasterize(
            const Eigen::Matrix<T, 4, 4>& projectionMatrix, 
            const Eigen::Matrix<T, 4, 4>& world2cam, 
            uint8_t* frameBuffer, 
            uint32_t width, 
            uint32_t height
        ) const override {
            // Transform vertices to clip space
            Eigen::Vector<T, 4> v0_clip = projectionMatrix * world2cam * Eigen::Vector<T, 4>(v0[0], v0[1], v0[2], 1.0);
            Eigen::Vector<T, 4> v1_clip = projectionMatrix * world2cam * Eigen::Vector<T, 4>(v1[0], v1[1], v1[2], 1.0);
            Eigen::Vector<T, 4> v2_clip = projectionMatrix * world2cam * Eigen::Vector<T, 4>(v2[0], v2[1], v2[2], 1.0);

            // Perspective divide to get NDC coordinates
            v0_clip /= v0_clip[3];
            v1_clip /= v1_clip[3]; 
            v2_clip /= v2_clip[3];

            // Convert NDC to screen coordinates
            Eigen::Vector<T, 2> v0_screen((v0_clip[0] + 1) * width * 0.5, (v0_clip[1] + 1) * height * 0.5);
            Eigen::Vector<T, 2> v1_screen((v1_clip[0] + 1) * width * 0.5, (v1_clip[1] + 1) * height * 0.5);
            Eigen::Vector<T, 2> v2_screen((v2_clip[0] + 1) * width * 0.5, (v2_clip[1] + 1) * height * 0.5);

            
            drawLine(v0_screen, v1_screen, frameBuffer, width, height);
            drawLine(v1_screen, v2_screen, frameBuffer, width, height);
            drawLine(v2_screen, v0_screen, frameBuffer, width, height);
        }
    };

    template <typename T>
    struct Light {
        virtual void sample(Eigen::Vector3<T>& p, Eigen::Vector3<T>& L_e, T& pdf) const = 0;
    };

    template <typename T>
    struct PointLight : Light<T> {
        Eigen::Vector3<T> position;
        Eigen::Vector3<T> intensity;
        T radius;

        /**
         * @brief Construct a new Point Light object
         * @param position The position of the point light
         * @param intensity The intensity of the point light
         * @param radius The radius of the point light
         */
        PointLight(const Eigen::Vector3<T>& position, const Eigen::Vector3<T>& intensity, T radius = 1) : position(position), intensity(intensity), radius(radius) {}

        /**
         * @brief Sample the point light
         */
        void sample(Eigen::Vector3<T>& p, Eigen::Vector3<T>& L_e, T& pdf) const override {
            if (radius == 0){
                p = position;
            }
            else{
                Eigen::Vector3<T> randomPoint = randomVector3<T>(Eigen::Vector3<T>::Constant(-1), Eigen::Vector3<T>::Constant(1));
                p = position + radius * randomPoint.normalized();
            }
            L_e = intensity;
            pdf = 1.0;
        }
    };

    template <typename T>
    struct HDRI : Light<T> {
        uint32_t numSamples;

        HDRI(uint32_t numSamples) : numSamples(numSamples) {}

        virtual void sample(Eigen::Vector3<T>& p, Eigen::Vector3<T>& L_e, T& pdf) const override {
            throw std::runtime_error("<HDRI> Sample function not implemented");
        }
    };

    template <typename T>
    struct ConstantHDRI : HDRI<T> {
        Eigen::Vector3<T> radiance;
        Eigen::Vector3<T> center;
        T radius;

        ConstantHDRI(
            const Eigen::Vector3<T>& radiance, 
            const Eigen::Vector3<T>& center = Eigen::Vector3<T>::Zero(), 
            T radius = 10000,
            uint32_t numSamples = 2
        ) : radiance(radiance), center(center), radius(radius), HDRI<T>(numSamples) {}

        void sample(Eigen::Vector3<T>& p, Eigen::Vector3<T>& L_e, T& pdf) const override {
            // Generate random spherical coordinates
            T theta = 2 * M_PI * randomReal<T>(0, 1);
            T phi = acos(randomReal<T>(-1, 1));
            
            // Convert to unit vector
            Eigen::Vector3<T> dir;
            dir[0] = sin(phi) * cos(theta);
            dir[1] = sin(phi) * sin(theta); 
            dir[2] = cos(phi);

            // Scale by radius and offset from center
            p = center + radius * dir;
            L_e = radiance;
            pdf = 1.0;
        }
    };
}
#endif
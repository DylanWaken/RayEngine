/**
 * @file utils.h
 * @author Dylan Sun
 * @brief Utility functions
 * @version 0.1
 * @date 2024-11-23
 */

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <random>
#include <thread>
#include <chrono>
namespace ray{

    /**
     * @brief Draw a line between two points on the frame buffer
     * @param p0 The start point
     * @param p1 The end point
     * @param frameBuffer The frame buffer (CHW)
     * @param width The width of the frame buffer
     * @param height The height of the frame buffer
     */
    template <typename T>
    void drawLine(const Eigen::Vector<T, 2>& p0, const Eigen::Vector<T, 2>& p1, uint8_t* frameBuffer, uint32_t width, uint32_t height,
        const Eigen::Vector<uint8_t, 3>& color = Eigen::Vector<uint8_t, 3>::Constant(255)){
        Eigen::Vector<T, 2> delta = p1 - p0;
        T step = std::max(std::abs(delta[0]), std::abs(delta[1]));
        delta /= step;

        Eigen::Vector<T, 2> current = p0;
        for (T i = 0; i <= step; ++i) {
            uint32_t x = static_cast<uint32_t>(current[0]);
            uint32_t y = static_cast<uint32_t>(current[1]);
            if (x >= width || y >= height) continue;
            if (x < 0 || y < 0) continue;
            frameBuffer[3 * (y * width + x)] = color[0];
            frameBuffer[3 * (y * width + x) + 1] = color[1];
            frameBuffer[3 * (y * width + x) + 2] = color[2];
            current += delta;
        }
    }

    /**
     * @brief Generate a random real number between min and max
     * @param min The minimum value
     * @param max The maximum value
     * @return A random real number
     */
    template <typename T>
    T randomReal(T min, T max){
        static thread_local std::mt19937 gen;
        std::uniform_real_distribution<T> dis(min, max);
        return dis(gen);
    }

    /**
     * @brief Generate a random vector3 between min and max
     * @param min The minimum value
     * @param max The maximum value
     * @return A random vector3
     */ 
    template <typename T>
    inline Eigen::Vector<T, 3> randomVector3(const Eigen::Vector<T, 3>& min, const Eigen::Vector<T, 3>& max){
        return Eigen::Vector<T, 3>(randomReal(min[0], max[0]), randomReal(min[1], max[1]), randomReal(min[2], max[2]));
    }

    /**
     * @brief Reflect a vector
     * @param wi The incoming vector
     * @param n The normal (assumed to be normalized)
     * @return The reflected vector
     */
    template <typename T>
    inline Eigen::Vector<T, 3> reflect(const Eigen::Vector<T, 3>& wi, const Eigen::Vector<T, 3>& n){
        return wi - 2 * n.dot(wi) * n;
    }

    uint32_t randomInt(uint32_t min, uint32_t max){
        static thread_local std::mt19937 gen;
        std::uniform_int_distribution<uint32_t> dis(min, max);
        return dis(gen);
    }

    /**
     * @brief Find the u forward direction of the triangle
     * @param v1 The first vertex
     * @param v2 The second vertex
     * @param v3 The third vertex
     * @param uv1 The first uv coordinate
     * @param uv2 The second uv coordinate
     * @param uv3 The third uv coordinate
     * @return The u forward direction
     */
    template <typename T>
    inline Eigen::Vector<T, 3> findUForward(
        const Eigen::Vector<T, 3>& v1,
        const Eigen::Vector<T, 3>& v2,
        const Eigen::Vector<T, 3>& v3,
        const Eigen::Vector<T, 2>& uv1,
        const Eigen::Vector<T, 2>& uv2,
        const Eigen::Vector<T, 2>& uv3
    ) {
        // Calculate edge vectors
        Eigen::Vector<T, 3> e1 = v2 - v1;
        Eigen::Vector<T, 3> e2 = v3 - v1;

        // Calculate UV differences
        T du1 = uv2[0] - uv1[0];
        T du2 = uv3[0] - uv1[0];
        T dv1 = uv2[1] - uv1[1];
        T dv2 = uv3[1] - uv1[1];

        // Calculate determinant
        T det = du1 * dv2 - du2 * dv1;
        
        // Handle degenerate case
        if (std::abs(det) < std::numeric_limits<T>::epsilon()) {
            // Return arbitrary perpendicular vector to normal in case of degenerate UV coordinates
            Eigen::Vector<T, 3> normal = e1.cross(e2).normalized();
            return normal.unitOrthogonal();
        }   

        // Calculate tangent vector
        T f = T(1) / det;
        Eigen::Vector<T, 3> tangent = (e1 * dv2 - e2 * dv1) * f;
        
        // Normalize the tangent vector
        return tangent.normalized();
    }

    /**
     * @brief Transform the shading normal to align with the geometry normal
     * @param shadingNormal The shading normal
     * @param geometryNormal The geometry normal
     * @param uForward The increasing direction of u in tangent space
     * @return The transformed shading normal
     */
    template <typename T>
    inline Eigen::Vector<T, 3> transformShadingNormal(
        const Eigen::Vector<T, 3>& shadingNormal,  // in the uv space 
        const Eigen::Vector<T, 3>& geometryNormal,
        const Eigen::Vector<T, 3>& uForward  
    ){
        Eigen::Vector<T, 3> normal = shadingNormal;
        if (normal.dot(geometryNormal) < 0){
            normal = -normal;
        }

        // Get the tangent vector in tangent space by finding component of uForward perpendicular to normal
        Eigen::Vector<T, 3> tangent = (uForward - uForward.dot(geometryNormal) * geometryNormal).normalized();

        // Get the bitangent vector in tangent space
        Eigen::Vector<T, 3> bitangent = tangent.cross(geometryNormal).normalized();

        // Create the rotation matrix
        Eigen::Matrix<T, 3, 3> rotationMatrix;
        rotationMatrix << tangent, bitangent, geometryNormal;

        // Transform the shading normal
        return rotationMatrix * normal;
    }
}
#endif
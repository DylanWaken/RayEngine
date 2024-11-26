/**
 * @file raySampler.h
 * @author Dylan Sun 
 * @brief Ray sampler for ray tracing, (sampling at camera / intersections)
 *        BRDF Monte Carlo Integration is Used
 * @version 0.1
 * @date 2024-11-23
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RAY_SAMPLER_H
#define RAY_SAMPLER_H

#pragma once

#include "rayTypes.h"
#include "parallelFor.h"
#include "utils.h"
#include <iostream>
namespace ray{
    /**
     * @brief Sample NDC space to generate rays
     * 
     * In Screen Space (NDC), all rays are in the form (px, py, 1, 1)
     * We throw them back to world space by multiplying the inverse of the projection matrix and the camera space to world space matrix
     * 
     * @param world2cam The world to camera transformation matrix
     * @param projMatrix The projection matrix
     * @param width The width of the image
     * @param height The height of the image
     * @param rays The vector of rays
     * @param nRays The number of rays to generate
     * @param jittered Whether to use jittered sampling (randomized samples within the pixel)
     */
    template <typename T>
    inline void sampleNDC(const Eigen::Matrix<T,4,4>& invWorld2Cam, const Eigen::Matrix<T,4,4>& invProjectionMatrix,
        uint32_t width, uint32_t height, std::vector<Ray<T>>& rays, uint32_t nRays, uint32_t AAsubdivisions){
        
        assert(AAsubdivisions > 0);
        assert(nRays % (AAsubdivisions * AAsubdivisions) == 0);
        uint32_t nRaysPerPixel = nRays / (AAsubdivisions * AAsubdivisions);

        // Generate width * height * nRays rays
        rays.reserve(width * height * nRays);
        for (uint32_t i = 0; i < width * height * nRays; ++i){
            rays.emplace_back();
        }

        // AA width and height
        uint32_t AAwidth = width * AAsubdivisions;
        uint32_t AAheight = height * AAsubdivisions;

        // Generate rays
        parallelFor(0, width * height, [&](uint32_t i){
            uint32_t x = i % width;
            uint32_t y = i / width;

            for (uint32_t k = 0; k < AAsubdivisions * AAsubdivisions; ++k){
                uint32_t subX = k % AAsubdivisions; 
                uint32_t subY = k / AAsubdivisions;
                for (uint32_t j = 0; j < nRaysPerPixel; ++j) {
                    // Compute normalized device coordinates (NDC)
                    T px = (2.0 * (x * AAsubdivisions + subX)) / AAwidth - 1.0;
                    T py = (2.0 * (y * AAsubdivisions + subY)) / AAheight - 1.0;  
                    // Create ray in NDC space
                    Eigen::Vector<T, 4> rayOrigin(px, py, 0, 1);
                    Eigen::Vector<T, 4> rayDirection(px, py, 1, 1);

                    // Transform to camera space
                    rayOrigin = invProjectionMatrix * rayOrigin;
                    rayDirection = invProjectionMatrix * rayDirection;

                    // Normalize w component
                    rayOrigin /= rayOrigin.w();
                    rayDirection /= rayDirection.w();

                    // Transform to world space
                    rayOrigin = invWorld2Cam * rayOrigin;
                    rayDirection = invWorld2Cam * rayDirection;

                    // Create final ray (normalize direction)
                    uint32_t rayIndex = i * nRays + j;
                    rays[rayIndex].origin = rayOrigin.template head<3>();
                    rays[rayIndex].direction = (rayDirection.template head<3>() - rayOrigin.template head<3>()).normalized();
                }
            }
        });
    }

    template <typename T>
    inline void sampleNDCThread(const Eigen::Matrix<T,4,4>& invWorld2Cam, const Eigen::Matrix<T,4,4>& invProjectionMatrix,
        uint32_t width, uint32_t height, std::vector<Ray<T>>& rays, uint32_t nRays, uint32_t AAsubdivisions,
        uint32_t startW, uint32_t startH){
          
        assert(AAsubdivisions > 0);
        assert(nRays % (AAsubdivisions * AAsubdivisions) == 0);
        uint32_t nRaysPerPixel = nRays / (AAsubdivisions * AAsubdivisions);

        // Generate width * height * nRays rays
        rays.reserve(nRays);
        for (uint32_t i = 0; i < nRays; ++i){
            rays.emplace_back();
        }

        // AA width and height
        uint32_t AAwidth = width * AAsubdivisions;
        uint32_t AAheight = height * AAsubdivisions;

        // Generate rays
        uint32_t x = startW;
        uint32_t y = startH;

        for (uint32_t k = 0; k < AAsubdivisions * AAsubdivisions; ++k){
            uint32_t subX = k % AAsubdivisions; 
            uint32_t subY = k / AAsubdivisions;
            for (uint32_t j = 0; j < nRaysPerPixel; ++j) {
                // Compute normalized device coordinates (NDC)
                T px = (2.0 * (x * AAsubdivisions + subX)) / AAwidth - 1.0;
                T py = (2.0 * (y * AAsubdivisions + subY)) / AAheight - 1.0;  // Flip y coordinate

                // Create ray in NDC space
                Eigen::Vector<T, 4> rayOrigin(px, py, 0, 1);
                Eigen::Vector<T, 4> rayDirection(px, py, 1, 1);

                // Transform to camera space
                rayOrigin = invProjectionMatrix * rayOrigin;
                rayDirection = invProjectionMatrix * rayDirection;

                // Normalize w component
                rayOrigin /= rayOrigin.w();
                rayDirection /= rayDirection.w();

                // Transform to world space
                rayOrigin = invWorld2Cam * rayOrigin;
                rayDirection = invWorld2Cam * rayDirection;

                // Create final ray (normalize direction)
                uint32_t rayIndex = j;
                rays[rayIndex].origin = rayOrigin.template head<3>();
                rays[rayIndex].direction = (rayDirection.template head<3>() - rayOrigin.template head<3>()).normalized();
            }
        }
    }

    /**
     * @brief Render NDC samples to framebuffer (for debugging)
     * 
     * @param rays The vector of rays
     * @param frameBuffer The framebuffer to render to
     * @param width The width of the image
     * @param height The height of the image
     */
    template <typename T>
    void renderNDC(const std::vector<Ray<T>>& rays, uint8_t* frameBuffer, uint32_t width, uint32_t height){
        parallelFor(0, width * height, [&](uint32_t i){
            uint32_t x = i % width;
            uint32_t y = i / width;
            uint32_t nRays = rays.size() / (width * height);

            T rp = 0, gp = 0, bp = 0;
            for (uint32_t j = 0; j < nRays; ++j){
                uint32_t rayIndex = i * nRays + j;
                const Ray<T>& ray = rays[rayIndex];
                
                // Get absolute direction components and scale to 0-255
                T r = std::abs(ray.direction[0]);
                T g = std::abs(ray.direction[1]); 
                T b = std::abs(ray.direction[2]);

                // Write to framebuffer
                rp += r;
                gp += g;
                bp += b;
            }
            frameBuffer[3 * (y * width + x)] = static_cast<uint8_t>(std::abs(rp / nRays * 255));
            frameBuffer[3 * (y * width + x) + 1] = static_cast<uint8_t>(std::abs(gp / nRays * 255));
            frameBuffer[3 * (y * width + x) + 2] = static_cast<uint8_t>(std::abs(bp / nRays * 255));
        });
    }

    /**
     * @brief Calculate the PDF of the cosine weighted hemisphere
     * 
     * PDF = cos(theta) / pi
     * 
     * @param n The normal
     * @param sample The incoming direction
     */
    template <typename T>
    inline T cosineHemispherePdf(const Eigen::Vector3<T>& n, const Eigen::Vector3<T>& sample){
        auto out = std::abs(n.dot(sample)) / M_PI;
        if (std::isnan(out)){
            std::cout<<"n: "<<n.transpose()<<std::endl;
            std::cout<<"sample: "<<sample.transpose()<<std::endl;
            std::cout<<"n.dot(sample): "<<n.dot(sample)<<std::endl;
            exit(0);
        }
        return out;
    }

    /**
     * @brief Sample the hemisphere using cosine weighted sampling
     * 
     * @param sample The sampled direction
     * @param pdf The pdf of the sampled direction
     * @param n The normal
     */
    template <typename T>
    inline void sampleCosineHemisphere(
        Eigen::Vector3<T>& sample, 
        T& pdf, 
        const Eigen::Vector3<T>& n
    ){
        // Get orthonormal basis aligned with normal
        Eigen::Vector3<T> b1, b2;
        if (std::abs(n.x()) > std::abs(n.y())) {
            b1 = Eigen::Vector3<T>(-n.z(), 0, n.x()).normalized();
        } else {
            b1 = Eigen::Vector3<T>(0, n.z(), -n.y()).normalized();
        }
        b2 = n.cross(b1);

        // Generate random points on unit disk using concentric mapping
        T r = std::sqrt(randomReal<T>(0, 1));
        T theta = 2 * M_PI * randomReal<T>(0, 1);
        T x = r * std::cos(theta);
        T y = r * std::sin(theta);

        // Project disk point onto hemisphere using cosine-weighted mapping
        T z = std::sqrt(std::max(T(0), 1 - x*x - y*y));

        // Transform sample to world space using basis vectors
        sample = x * b1 + y * b2 + z * n;

        // PDF for cosine weighted sampling is cos(theta)/pi
        pdf = cosineHemispherePdf(n, sample);
    }

    /**
     * @brief Calculate the PDF of the Phong Specular BRDF
     * 
     * PDF = (exponent + 1) * (wi dot sample) ^ exponent / (2 * pi)
     * 
     * @param wo The outgoing direction
     * @param sample The sampled direction
     * @param n The normal
     * @param exponent The exponent of the Phong BRDF
     */
    template <typename T>
    inline T phongSpecularPdf(const Eigen::Vector3<T>& wo, const Eigen::Vector3<T>& sample, const Eigen::Vector3<T>& n, const T exponent){
        Eigen::Vector3<T> wi = reflect(wo, n);
        auto out = (exponent + 1) * std::pow(std::abs(sample.dot(wi)), exponent) / (2 * M_PI);
        return out;
    }

    /**
     * @brief Sample the Phong Specular BRDF
     * 
     * Phong BRDF is defined as k * (wi dot wo) ^ exponent
     * 
     * @param sample The sampled direction
     * @param pdf The pdf of the sampled direction
     * @param wo The outgoing direction
     * @param n The normal
     * @param exponent The exponent of the Phong BRDF
     */
    template <typename T>
    inline void samplePhongSpecular(
        Eigen::Vector3<T>& sample, 
        T& pdf, 
        const Eigen::Vector3<T>& wo, 
        const Eigen::Vector3<T>& n, 
        const T exponent
    ){
        // Get reflected direction
        Eigen::Vector3<T> wi = reflect(wo, n);
        wi.normalize();

        // Create orthonormal basis around reflected direction
        Eigen::Vector3<T> b1;
        if (std::abs(wi.x()) < std::abs(wi.y())) {
            b1 = Eigen::Vector3<T>(0, -wi.z(), wi.y()).normalized();
        } else {
            b1 = Eigen::Vector3<T>(-wi.z(), 0, wi.x()).normalized();
        }
        Eigen::Vector3<T> b2 = wi.cross(b1);

        // Sample direction using Phong importance sampling
        T u1 = randomReal<T>(0, 1);
        T u2 = randomReal<T>(0, 1);
        
        // Convert uniform random numbers to Phong distribution
        T cosTheta = std::pow(u1, 1 / (exponent + 1));
        T sinTheta = std::sqrt(1 - cosTheta * cosTheta);
        T phi = 2 * M_PI * u2;

        // Calculate directions in local space
        T x = sinTheta * std::cos(phi);
        T y = sinTheta * std::sin(phi);
        T z = cosTheta;

        // Transform to world space using basis vectors
        sample = x * b1 + y * b2 + z * wi;
        sample.normalize();

        // Calculate PDF values
        pdf = phongSpecularPdf(wo, sample, n, exponent);
    }
}

#endif
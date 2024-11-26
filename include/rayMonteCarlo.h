/**
 * @file rayMonteCarlo.h
 * @author Dylan Sun
 * @brief Monte Carlo integration for Geometric Optics Light Transport (LTE)
 * @version 0.1
 * @date 2024-11-23
 */

#ifndef RAY_MONTE_CARLO_H
#define RAY_MONTE_CARLO_H

#include "principleBRDF.h"
#include "rayTypes.h"
#include "intersectPrimitives.h"
#include <vector>
#include <stack>
namespace ray{
    /**
     * @brief Monte Carlo Light Transport Equation
     * 
     * f(p2 -> p1 -> p0) | cos(theta)_1 | / P(dir(p0, p1))
     * 
     * @param wo The outgoing direction p1 -> p0
     * @param p1 The intersection point OF EVALUATION
     * @param wi The incoming direction p2 -> p1
     * @param primitives The primitives in the scene
     * @param pdfP1 The pdf of the intersection point
     * @return The emitted radiance
     */
    template <typename T>
    inline Eigen::Vector3<T> monteCarloLTE(
        const Eigen::Vector3<T>& wo, 
        const Intersection<T>& p1, 
        const Eigen::Vector3<T>& wi,
        const std::vector<Primitive<T>*>& primitives,
        const T& pdfP1
    ){
        return primitives[p1.primitiveID]->brdf->eval(wo, p1.normal, wi, p1.tangent, p1.uv) * std::abs(wi.dot(p1.normal)) / pdfP1;
    }

    /**
     * @brief Monte Carlo Light Transport Equation for Final Direct Lighting
     * 
     * @param wo The outgoing direction p1 -> p0
     * @param p1 The intersection point OF EVALUATION
     * @param primitives The primitives in the scene
     * @param bvhNodes The BVH nodes in the scene
     * @param lights The lights in the scene
     * @param numLightSamples The number of light samples
     * @return The emitted radiance
     */
    template <typename T>
    inline Eigen::Vector3<T> monteCarloLights(
        const Eigen::Vector3<T>& wo, 
        const Intersection<T>& p1, 
        const std::vector<Primitive<T>*>& primitives,
        const std::vector<BVHNode<T>>& bvhNodes,
        const std::vector<Light<T>*>& lights,
        const HDRI<T>* hdriLight,
        const uint32_t& numLightSamples
    ){
        Eigen::Vector3<T> L = Eigen::Vector3<T>::Zero();

        // Sample each light source
        for (uint32_t i = 0; i < numLightSamples; ++i) {
            // Randomly select a light
            uint32_t hdriSamples = 0;
            if (hdriLight){
                hdriSamples = hdriLight->numSamples;
            }

            uint32_t lightIndex = randomInt(0, lights.size() - 1 + hdriSamples);

            Eigen::Vector3<T> lightPos;
            Eigen::Vector3<T> Le;
            T pdf;
            T dist;

            if (lightIndex < lights.size()){
                Light<T>* light = lights[lightIndex];
                // Sample the light
                light->sample(lightPos, Le, pdf);
                dist = (lightPos - p1.position).norm();
            } else {
                hdriLight->sample(lightPos, Le, pdf);
                dist = 1;
            }

            if (!visible(p1.position, lightPos, primitives, bvhNodes, p1.primitiveID)){
                continue;
            }

            // Calculate direction from intersection to light
            Eigen::Vector3<T> wi = (lightPos - p1.position).normalized();
            
            // Calculate geometric term
            T cosTheta = std::abs(wi.dot(p1.normal));

            // Evaluate BRDF at intersection point
            Eigen::Vector3<T> brdf = primitives[p1.primitiveID]->brdf->eval(
                wo, 
                p1.normal, 
                wi, 
                p1.tangent,
                p1.uv
            );

            // Add contribution using Monte Carlo estimator
            // Le * brdf * cos / (dist^2 * pdf)
            L += Le.cwiseProduct(brdf) * cosTheta / (dist * dist * pdf);
        }

        // Average over number of samples
        return L / numLightSamples;
    }



    template <typename T>
    inline Eigen::Vector3<T> pathTracing(
        const Ray<T>& ray,
        const std::vector<Primitive<T>*>& primitives,
        const std::vector<BVHNode<T>>& bvhNodes,
        const std::vector<Light<T>*>& lights,
        const HDRI<T>* hdriLight,
        const uint32_t& numLightSamples = 10,
        const uint32_t& maxDepth = 3
    ) {
        // Initialize accumulated color and throughput
        Eigen::Vector3<T> accumulatedColor = Eigen::Vector3<T>::Zero();
        Eigen::Vector3<T> throughput = Eigen::Vector3<T>::Ones();
        
        // Current ray state
        Ray<T> currentRay = ray;
        uint32_t depth = 0;
        
        while (depth < maxDepth) {
            // Find intersection
            Intersection<T> intersection = trace(currentRay, primitives, bvhNodes);
            
            // If no hit, break the path
            if (!intersection.hit) {
                break;
            }

            // Apply Russian Roulette
            if (depth > 2) {
                // Survival probability based on throughput luminance
                T p = std::min(std::max(throughput.maxCoeff(), T(0.1)), T(0.95));
                
                // Random number between 0 and 1
                if (randomReal<T>(0, 1) > p) {
                    break;  // Terminate the path
                }
                // Scale throughput to compensate for terminated paths
                throughput /= p;
            }
            
            // Store the negated direction first
            Eigen::Vector3<T> wo = -currentRay.direction;
            Eigen::Vector3<T> directLight = monteCarloLights(
                wo,  // Pass the stored vector instead of the temporary
                intersection,
                primitives,
                bvhNodes,
                lights,
                hdriLight,
                numLightSamples
            );

            accumulatedColor += directLight.cwiseProduct(throughput);
            
            // Sample next direction for indirect lighting
            Eigen::Vector3<T> wi;
            T pdf;

            // Sample the BRDF
            primitives[intersection.primitiveID]->brdf->sample(
                wi,
                pdf,
                wo,
                intersection.normal,
                intersection.tangent,
                intersection.uv
            );
            
            // Calculate BRDF contribution for indirect bounce
            Eigen::Vector3<T> brdfContribution = monteCarloLTE(
                wo,
                intersection,
                wi,
                primitives,
                pdf
            );
            
            // Update throughput for next bounce
            throughput = throughput.cwiseProduct(brdfContribution);
            
            // Set up next ray
            currentRay = Ray<T>(
                intersection.position, 
                wi
            );
            currentRay.leavingPrimitiveID = intersection.primitiveID;
            depth++;
        }
    
        return accumulatedColor;
    }
}

#endif
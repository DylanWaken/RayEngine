/**
 * @file rayEngine.h
 * @author Dylan Sun 
 * @brief BRDF for principle materials
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef PRINCIPLEBRDF_H
#define PRINCIPLEBRDF_H

#pragma once

#include "utils.h"
#include "raySampler.h"
#include <vector>
#include <cassert>
#include "pbrMaps.h"
#include <iostream>
namespace ray{
    template <typename T>
    struct BRDF{
        PBRMap<T>* pbrMap;

        /**
         * @brief Constructor
         * @param pbrMap The PBR map
         */
        BRDF(PBRMap<T>* pbrMap) : pbrMap(pbrMap) {}

        /**
         * @brief Sample the BRDF using Multiple Importance Sampling (MIS)
         * @param wo The outgoing direction
         * @param n The normal (geometry normal), will use shading normal in SVBRDF adjusted by geometry normal
         * @param uv The uv coordinates
         * @param samples The sampled directions
         * @param pdfs The pdfs of the sampled directions
         */
        virtual void sample(
            Eigen::Vector3<T>& sample,
            T& pdf,
            const Eigen::Vector3<T>& wo, 
            const Eigen::Vector3<T>& n, 
            const Eigen::Vector3<T>& tangent,
            const Eigen::Vector2<T>& uv,
            bool useShadingNormal = false
        ) const = 0;

        /**
         * @brief Evaluate the BRDF
         * @param wo The outgoing direction
         * @param wi The incoming direction
         * @param tangent The tangent vector (from tangent space)
         * @return The BRDF value
         */
        virtual Eigen::Vector3<T> eval(
            const Eigen::Vector3<T>& wo, 
            const Eigen::Vector3<T>& n, 
            const Eigen::Vector3<T>& wi, 
            const Eigen::Vector3<T>& tangent,
            const Eigen::Vector2<T>& uv,
            bool useShadingNormal = false
        ) const = 0;
    };

    /**
     * @brief Specular Lambertian BRDF
     * 
     * A simple Lambertian BRDF with a Phong specular component
     */
    template <typename T>
    struct SpecularLambertianBRDF : public BRDF<T>{

        T diffuseWeightMultiplier = 1;
        T specularWeightMultiplier = 1;
        /**
         * @brief Constructor
         * @param pbrMap The PBR map
         */
        SpecularLambertianBRDF(
            PBRMap<T>* pbrMap, 
            T diffuseWeightMultiplier = 1,
            T specularWeightMultiplier = 1) :
        BRDF<T>(pbrMap), diffuseWeightMultiplier(diffuseWeightMultiplier), specularWeightMultiplier(specularWeightMultiplier) {}

        /**
         * @brief Sample the BRDF
         * @param wo The outgoing direction
         * @param n The normal (geometry normal), will use shading normal in SVBRDF adjusted by geometry normal
         * @param uv The uv coordinates (not used in this BRDF)
         * @return A possible incoming direction
         */
        virtual void sample(
            Eigen::Vector3<T>& sample,
            T& pdf,
            const Eigen::Vector3<T>& wo, 
            const Eigen::Vector3<T>& n, 
            const Eigen::Vector3<T>& tangent,
            const Eigen::Vector2<T>& uv,
            bool useShadingNormal = false
        ) const override{
            // Get the PBR map values with epsilon protection
            T specularExponent = std::max(this->pbrMap->specularExponent(uv), T(1e-5));
            Eigen::Vector3<T> specularColor = this->pbrMap->specularColor(uv);
            Eigen::Vector3<T> baseColor = this->pbrMap->baseColor(uv);

            // Ensure colors are valid
            specularColor = specularColor.cwiseMax(Eigen::Vector3<T>::Constant(1e-5));
            baseColor = baseColor.cwiseMax(Eigen::Vector3<T>::Constant(1e-5));

            Eigen::Vector3<T> normal;
            if (useShadingNormal) {
                normal = this->pbrMap->normal(uv);
                normal = transformShadingNormal(normal, n, tangent);
            } else {
                normal = n;
            }

            // Ensure normal is normalized
            normal.normalize();

            // Ensure wo and normal are in the same hemisphere
            if (wo.dot(normal) <= 0){
                normal = -normal;
            }

            // Calculate weights with protection against zero
            T specularWeight = std::max(specularColor.mean(), T(1e-5)) * specularWeightMultiplier;
            T diffuseWeight = std::max(baseColor.mean(), T(1e-5)) * diffuseWeightMultiplier;
            T totalWeight = specularWeight + diffuseWeight;

            // Safe normalization of weights
            specularWeight = specularWeight / totalWeight;
            diffuseWeight = diffuseWeight / totalWeight;


            SAMPLE:
            // Random choice between components
            if (randomReal<T>(0, 1) < specularWeight) {
                // Sample from specular component
                samplePhongSpecular(sample, pdf, wo, normal, specularExponent);
                
                // Protect against zero PDFs
                T specularPdf = std::max(pdf, T(1e-5));
                T diffusePdf = std::max(cosineHemispherePdf(normal, sample), T(1e-5));
                
                // Safe MIS weight calculation
                T denominator = specularWeight * specularPdf + diffuseWeight * diffusePdf;
                if (denominator > T(1e-5)) {
                    pdf = (specularWeight * specularPdf) / denominator;
                } else {
                    pdf = T(2.0); // Fallback value
                }
            } else {
                // Sample from diffuse component
                sampleCosineHemisphere(sample, pdf, normal);

                // Protect against zero PDFs
                T diffusePdf = std::max(pdf, T(1e-5));
                T specularPdf = std::max(phongSpecularPdf(wo, sample, normal, specularExponent), T(1e-5));
                
                // Safe MIS weight calculation
                T denominator = specularWeight * specularPdf + diffuseWeight * diffusePdf;
                if (denominator > T(1e-5)) {
                    pdf = (diffuseWeight * diffusePdf) / denominator;
                } else {
                    pdf = T(3.0); // Fallback value
                }
            }

            // Ensure sample and normal are in the same hemisphere (this is not an BTDF)
            if (sample.dot(normal) <= 0){
                goto SAMPLE;
            }

            // Final safety check
            if (std::isnan(pdf) || pdf <= T(0)) {
                pdf = T(1.0); // Fallback value
                sample = normal; // Default to normal direction
            }
        }

        /**
         * @brief Evaluate the BRDF
         * @param wo The outgoing direction
         * @param n The normal (geometry normal), will use shading normal in SVBRDF adjusted by geometry normal
         * @param wi The incoming direction
         * @param uv The uv coordinates 
         * @return The BRDF value
         */
        virtual Eigen::Vector3<T> eval(
            const Eigen::Vector3<T>& wo, 
            const Eigen::Vector3<T>& n, 
            const Eigen::Vector3<T>& wi, 
            const Eigen::Vector3<T>& tangent,
            const Eigen::Vector2<T>& uv,
            bool useShadingNormal = false
        ) const override {
            // Get the PBR map values with epsilon protection
            T specularExponent = std::max(this->pbrMap->specularExponent(uv), T(1e-5));
            Eigen::Vector3<T> specularColor = this->pbrMap->specularColor(uv).cwiseMax(Eigen::Vector3<T>::Constant(1e-5));
            Eigen::Vector3<T> baseColor = this->pbrMap->baseColor(uv).cwiseMax(Eigen::Vector3<T>::Constant(1e-5));

            Eigen::Vector3<T> normal;
            if (useShadingNormal) {
                normal = this->pbrMap->normal(uv);
                normal = transformShadingNormal(normal, n, tangent);
            } else {
                normal = n;
            }
            normal.normalize();

            // Early exit if no light contribution
            T NdotWi = wi.dot(normal);
            T NdotWo = wo.dot(normal);
            if (NdotWi <= 0 && NdotWo <= 0) {
                normal = -normal;
            } else if (NdotWi <= 0 || NdotWo <= 0) {
                return Eigen::Vector3<T>::Zero();
            }

            // Calculate specular component
            Eigen::Vector3<T> h = (wi + wo).normalized();
            T cosAlpha = std::max(h.dot(normal), T(0));
            T specularComponent = std::pow(cosAlpha, specularExponent);

            // Calculate diffuse component with energy conservation
            T diffuseComponent = T(1) / M_PI;

            // Combine components with energy conservation
            T specularWeight = std::max(specularColor.mean(), T(1e-5));
            T diffuseWeight = std::max(baseColor.mean(), T(1e-5));
            T totalWeight = specularWeight + diffuseWeight;

            specularWeight /= totalWeight;
            diffuseWeight /= totalWeight;

            return (specularComponent * specularColor * specularWeight + 
                    diffuseComponent * baseColor * diffuseWeight);
        }
    };

    /**
     * @brief Principle BRDF
     * 
     * A principle BRDF with a base color, roughness, and metallic component
     * Microfacet distribution model
     */
    template <typename T>
    struct PrincipleBRDF : public BRDF<T>{
        Eigen::Vector3<T> baseColor;
        T roughness;
        T metallic;
    };
}


#endif
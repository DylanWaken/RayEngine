/**
 * @file pbrMaps.h
 * @author Dylan Sun
 * @brief PBR maps
 * @version 0.1
 * @date 2024-11-24
 */
#pragma once

#include "utils.h"

namespace ray{
    template <typename T>
    struct PBRMap{
        /**
         * @brief Get the base color of the PBR map
         * @param uv The uv coordinates
         * @return The base color
         */
        virtual Eigen::Vector3<T> baseColor(const Eigen::Vector2<T>& uv) const = 0;

        /**
         * @brief Get the roughness of the PBR map
         * @param uv The uv coordinates
         * @return The roughness
         */
        virtual T roughness(const Eigen::Vector2<T>& uv) const = 0;

        /**
         * @brief Get the metallic of the PBR map
         * @param uv The uv coordinates
         * @return The metallic
         */
        virtual T metallic(const Eigen::Vector2<T>& uv) const = 0;

        /**
         * @brief Get the light map of the PBR map
         * @param uv The uv coordinates
         * @return The light map
         */
        virtual Eigen::Vector3<T> light(const Eigen::Vector2<T>& uv) const = 0;
        
        /**
         * @brief Get the normal map of the PBR map
         * @param uv The uv coordinates
         * @return The normal map
         */
        virtual Eigen::Vector3<T> normal(const Eigen::Vector2<T>& uv) const = 0;

        /**
         * @brief Get the specular color of the PBR map
         * @param uv The uv coordinates
         * @return The specular color
         */
        virtual Eigen::Vector3<T> specularColor(const Eigen::Vector2<T>& uv) const = 0;

        /**
         * @brief Get the specular exponent of the PBR map
         * @param uv The uv coordinates
         * @return The specular exponent
         */
        virtual T specularExponent(const Eigen::Vector2<T>& uv) const = 0;
    };

    template <typename T>
    struct ConstantSpecularLambertianPBRMap : public PBRMap<T>{
        Eigen::Vector3<T> bc, sc;
        T se;

        // Constructor with parameters
        ConstantSpecularLambertianPBRMap(const Eigen::Vector3<T>& bc, const Eigen::Vector3<T>& sc, const T& se) : bc(bc), sc(sc), se(se) {}

        virtual Eigen::Vector3<T> baseColor(const Eigen::Vector2<T>& uv) const override{
            return bc;
        }

        virtual T roughness(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual T metallic(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual Eigen::Vector3<T> light(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual Eigen::Vector3<T> normal(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual Eigen::Vector3<T> specularColor(const Eigen::Vector2<T>& uv) const override{
            return sc;
        }

        virtual T specularExponent(const Eigen::Vector2<T>& uv) const override{
            return se;
        }
    };


    template <typename T>
    struct VertexSpecularLambertianPBRMap : public PBRMap<T>{
        // Base color
        Eigen::Vector3<T> bc1, bc2, bc3;
        // Specular color
        Eigen::Vector3<T> sc1, sc2, sc3;
        // Specular exponent
        T se1, se2, se3;
        // default uvs: (0, 0), (1, 0), (0, 1)

        // Default constructor
        VertexSpecularLambertianPBRMap() = default;

        // Constructor with parameters
        VertexSpecularLambertianPBRMap(const Eigen::Vector3<T>& bc1, const Eigen::Vector3<T>& bc2, const Eigen::Vector3<T>& bc3,
                                       const Eigen::Vector3<T>& sc1, const Eigen::Vector3<T>& sc2, const Eigen::Vector3<T>& sc3,
                                       const T& se1, const T& se2, const T& se3) : 
                                       bc1(bc1), bc2(bc2), bc3(bc3), sc1(sc1), sc2(sc2), sc3(sc3), se1(se1), se2(se2), se3(se3) {}
    
        virtual Eigen::Vector3<T> baseColor(const Eigen::Vector2<T>& uv) const override{
            return bc1 * (1 - uv.x() - uv.y()) + bc2 * uv.x() + bc3 * uv.y();
        }

        virtual T roughness(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual T metallic(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual Eigen::Vector3<T> light(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual Eigen::Vector3<T> normal(const Eigen::Vector2<T>& uv) const override{
            throw std::runtime_error("Not implemented");
        }

        virtual Eigen::Vector3<T> specularColor(const Eigen::Vector2<T>& uv) const override{
            return sc1 * (1 - uv.x() - uv.y()) + sc2 * uv.x() + sc3 * uv.y();
        }

        virtual T specularExponent(const Eigen::Vector2<T>& uv) const override{
            return se1 * (1 - uv.x() - uv.y()) + se2 * uv.x() + se3 * uv.y();
        }
    };
}
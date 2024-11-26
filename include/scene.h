/**
 * @file rayEngine.h
 * @author Dylan Sun 
 * @brief Scene structure for ray tracing
 * @version 0.1
 * @date 2024-11-18
 * 
 * @copyright Copyright (c) 2024
 */
#ifndef SCENE_H
#define SCENE_H

#pragma once

#include "rayTypes.h"
#include "intersectPrimitives.h"
#include "bvh.h"
#include "parallelFor.h"
#include "raySampler.h"
#include "raySolver.h"
#include "rayMonteCarlo.h"
#define DEGREE_TO_RADIAN(degree) (degree * M_PI / 180.0)
#define MIN_DEPTH 0.0001

namespace ray{

    enum class RenderMode : uint8_t {
        PRIMITIVE = 0,  
        BVH = 1,
        NDC_SAMPLES = 2,
        NORMAL = 3,
        TANGENT = 4,
        UV = 5,
        DEPTH = 6,
        BASE_COLOR = 7,
        SHADING_NORMAL = 8,
        DIRECT_LIGHTING = 9,
        PATH_TRACING = 10,
    };

    /**
     * @brief Scene structure for ray tracing
     * @tparam T The type of the primitives
     */
    template <typename T>
    struct Scene{
        std::vector<Primitive<T>*> primitives;
        std::vector<BVHNode<T>> bvhNodes;
        std::vector<BRDF<T>*> brdfs;
        std::vector<Light<T>*> lights;
        HDRI<T>* hdriLight;

        // Framebuffer
        unsigned char* frameBuffer;
        uint32_t width, height;
        Eigen::Vector3<T> unRenderedColor = Eigen::Vector3<T>::Ones() * 0.2;

        // Sampling parameters
        uint32_t nPixelSamples = 1024; // rays per pixel, need to be divisible by AAsubdivisions * AAsubdivisions
        uint32_t AAsubdivisions = 2; // subpixels per pixel for antialiasing
        uint32_t maxPathDepth = 5; // max depth of the path tracing
        uint32_t nLightSamples = 10; // number of light samples for direct lighting per step

        // rendering are done in a block-wise manner
        uint32_t blockW = 32;
        uint32_t blockH = 32;

        // Projection matrix, only for debug rasterization.
        Eigen::Matrix<T, 4, 4> projectionMatrix;
        Eigen::Matrix<T, 4, 4> invProjectionMatrix; 
        // Transformation matrices (its inverse is the camera to world matrix)
        Eigen::Matrix<T, 4, 4> world2cam;
        Eigen::Matrix<T, 4, 4> invWorld2Cam;

        // Render mode  
        RenderMode renderMode = RenderMode::NORMAL;

        // Thread for rendering
        std::thread rendererThread;

        bool primitivesSet = false;
        bool bvhBuilt = false;
        bool isRendererRunning = false;
        bool isRendererFinished = false;

        /**
         * @brief Set the view matrix
         * @param eye The eye position
         * @param focus The focus position
         * @param up The up vector
         */
        inline void lookAt(const Eigen::Vector3<T>& eye = Eigen::Vector3<T>::UnitZ() * (8), 
            const Eigen::Vector3<T>& focus = Eigen::Vector3<T>::Zero(), 
            const Eigen::Vector3<T>& up = Eigen::Vector3<T>::UnitY()){
            // Calculate the forward, right and up vectors
            Eigen::Vector3<T> forward = (focus - eye).normalized();
            Eigen::Vector3<T> right = forward.cross(up).normalized();
            Eigen::Vector3<T> upVec = right.cross(forward);

            // Construct view matrix
            world2cam = Eigen::Matrix<T, 4, 4>::Identity();

            // First row - right vector and translation
            world2cam(0,0) = right.x();
            world2cam(0,1) = right.y(); 
            world2cam(0,2) = right.z();
            world2cam(0,3) = -right.dot(eye);

            // Second row - up vector and translation
            world2cam(1,0) = upVec.x();
            world2cam(1,1) = upVec.y();
            world2cam(1,2) = upVec.z(); 
            world2cam(1,3) = -upVec.dot(eye);

            // Third row - forward vector and translation
            world2cam(2,0) = -forward.x();
            world2cam(2,1) = -forward.y();
            world2cam(2,2) = -forward.z();
            world2cam(2,3) = forward.dot(eye);

            // Fourth row
            world2cam(3,0) = 0;
            world2cam(3,1) = 0;
            world2cam(3,2) = 0;
            world2cam(3,3) = 1;

            invWorld2Cam = world2cam.inverse();

            isRendererFinished = false;
        }

        /**
         * @brief Set the projection matrix
         * @param fov The field of view
         * @param aspectRatio The aspect ratio
         * @param near The near plane
         * @param far The far plane
         */
        inline void setProjectionMatrix(T fov = 60.0, T aspectRatio = 1920.0 / 1080.0, T near = 0.01, T far = 100.0){
            fov = DEGREE_TO_RADIAN(fov);
            T tanHalfFov = std::tan(fov * 0.5);
            T f = 1.0 / tanHalfFov;
            
            projectionMatrix = Eigen::Matrix<T, 4, 4>::Zero();
            projectionMatrix(0,0) = f / aspectRatio;
            projectionMatrix(1,1) = f;
            projectionMatrix(2,2) = -(far + near) / (far - near);
            projectionMatrix(2,3) = -2.0 * far * near / (far - near);
            projectionMatrix(3,2) = -1.0;

            invProjectionMatrix = projectionMatrix.inverse();

            isRendererFinished = false;
        }

        inline void initFrameBuffer(uint32_t width, uint32_t height){
            frameBuffer = new unsigned char[3 * width * height];
            this->width = width;
            this->height = height;
            parallelFor(0, width * height, [&](uint32_t i){
                uint32_t pixelIndex = i * 3;
                frameBuffer[pixelIndex] = unRenderedColor[0] * 255;
                frameBuffer[pixelIndex + 1] = unRenderedColor[1] * 255;
                frameBuffer[pixelIndex + 2] = unRenderedColor[2] * 255;
            });

            isRendererFinished = false;
        }

        template<typename... Args>
        inline void addTrianglePrimitive(Args&&... args) {
            primitives.push_back(new Triangle<T>(std::forward<Args>(args)...));
        }

        template<typename... Args>
        inline void addSpherePrimitive(Args&&... args) {
            primitives.push_back(new Sphere<T>(std::forward<Args>(args)...));
        }

        template<typename... Args>
        inline void addPointLight(Args&&... args) {
            lights.push_back(new PointLight<T>(std::forward<Args>(args)...));
        }

        inline void bindBRDF(uint32_t primitiveID, BRDF<T>* brdf){
            primitives[primitiveID]->brdf = brdf;
            brdfs.push_back(brdf);
        }

        inline void bindHDRI(HDRI<T>* hdriLight){
            this->hdriLight = hdriLight;
        }

        inline void finishPrimitivesSetup(){
            primitivesSet = true;
            std::cout << "<RayEngine> Primitives set with " << primitives.size() << " primitives." << std::endl;
            // Find max and min coordinates across all primitives
            if (primitives.empty()) return;
            
            for (size_t i = 0; i < primitives.size(); i++){
                primitives[i]->primitiveID = i;
            }

            Eigen::Vector3<T> minCoords = primitives[0]->getAABB().min;
            Eigen::Vector3<T> maxCoords = primitives[0]->getAABB().max;

            for (size_t i = 1; i < primitives.size(); i++) {
                const AABB<T>& bounds = primitives[i]->getAABB();
                minCoords = minCoords.cwiseMin(bounds.min);
                maxCoords = maxCoords.cwiseMax(bounds.max);
            }

            std::cout << "<RayEngine> Scene bounds: " << std::endl;
            std::cout << "Min: (" << minCoords.x() << ", " << minCoords.y() << ", " << minCoords.z() << ")" << std::endl;
            std::cout << "Max: (" << maxCoords.x() << ", " << maxCoords.y() << ", " << maxCoords.z() << ")" << std::endl;
        
            isRendererFinished = false;
        }

        /**
         * @brief Initialize the BVH
         */
        inline void initBVH(){
            buildBinaryBVH(primitives, bvhNodes);
            bvhBuilt = true;

            isRendererFinished = false;
        }

        void onResizeCanvas(uint32_t width, uint32_t height){
            if (isRendererRunning){
                std::cout << "<RayEngine> Renderer threadis running. Resizing canvas while rendering is not supported." << std::endl;
                return;
            }
            
            delete[] frameBuffer;
            initFrameBuffer(width, height);
            this->width = width;
            this->height = height;

            // Start new render
            if (isRendererRunning){
                rendererThread = std::thread(&Scene::onRenderThread, this);
            }

            isRendererFinished = false;
        }

        void onRender(){
            if (isRendererRunning){
                return;
            }
            if (!isRendererFinished){    
                isRendererRunning = true;
                rendererThread = std::thread(&Scene::onRenderThread, this);
                rendererThread.detach(); // Detach thread to let main thread continue
            }
        }

        void onRenderThread(){
            // TODO: Implement the rendering
            if (renderMode == RenderMode::PRIMITIVE){
                if (!primitivesSet){
                    std::cout << "<RayEngine> Primitives not set. skipping rendering." << std::endl;
                    return;
                }
                parallelFor(0, primitives.size(), [&](uint32_t i){
                    primitives[i]->projectAndRasterize(projectionMatrix, world2cam, frameBuffer, width, height);
                });
            }

            else if (renderMode == RenderMode::BVH){
                // TODO: Implement the BVH rendering
                if (!bvhBuilt){
                    std::cout << "<RayEngine> BVH not built. skipping rendering." << std::endl;
                    return;
                }
                parallelFor(0, primitives.size(), [&](uint32_t i){
                    primitives[i]->projectAndRasterize(projectionMatrix, world2cam, frameBuffer, width, height);
                });
                parallelFor(0, bvhNodes.size(), [&](uint32_t i){
                    bvhNodes[i].aabb.projectAndRasterize(projectionMatrix, world2cam, frameBuffer, width, height);
                });
            }

            else if (renderMode == RenderMode::NDC_SAMPLES){
                std::vector<Ray<T>> rays;
                sampleNDC(invWorld2Cam, invProjectionMatrix, width, height, rays, 1, 1);
                renderNDC(rays, frameBuffer, width, height);
            }

            else if (renderMode == RenderMode::NORMAL){
                // Render the normal of the hit point
                std::vector<Ray<T>> rays;
                sampleNDC(invWorld2Cam, invProjectionMatrix, width, height, rays, 1, 1);
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    Eigen::Vector3<T> color = Eigen::Vector3<T>::Zero();
                    Intersection<T> intersection = trace(rays[y * width + x], primitives, bvhNodes);
                    if (intersection.hit){
                        Eigen::Vector3<T> normal = intersection.normal;
                        // Convert normal from [-1,1] to [0,1] range and map to RGB
                        color = (normal + Eigen::Vector3<T>::Ones()) * 0.5;
                    }
                    uint32_t pixelIndex = (y * width + x) * 3;
                    frameBuffer[pixelIndex] = static_cast<uint8_t>(color[0] * 255);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>(color[1] * 255); 
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>(color[2] * 255);
                });
            }

            else if (renderMode == RenderMode::TANGENT){
                // Render the tangent of the hit point
                std::vector<Ray<T>> rays;
                sampleNDC(invWorld2Cam, invProjectionMatrix, width, height, rays, 1, 1);
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    Eigen::Vector3<T> color = Eigen::Vector3<T>::Zero();
                    Intersection<T> intersection = trace(rays[y * width + x], primitives, bvhNodes);
                    if (intersection.hit){
                        Eigen::Vector3<T> tangent = intersection.tangent;
                        // Convert tangent from [-1,1] to [0,1] range and map to RGB
                        color = (tangent + Eigen::Vector3<T>::Ones()) * 0.5;
                    }
                    uint32_t pixelIndex = (y * width + x) * 3;
                    frameBuffer[pixelIndex] = static_cast<uint8_t>(color[0] * 255);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>(color[1] * 255); 
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>(color[2] * 255);
                });
            }

            else if (renderMode == RenderMode::UV){
                // Render the uv of the hit point
                std::vector<Ray<T>> rays;
                sampleNDC(invWorld2Cam, invProjectionMatrix, width, height, rays, 1, 1);
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    Eigen::Vector3<T> color = Eigen::Vector3<T>::Zero();
                    Intersection<T> intersection = trace(rays[y * width + x], primitives, bvhNodes);
                    if (intersection.hit){
                        Eigen::Vector2<T> uv = intersection.uv;
                        // Map UV coordinates directly to RG channels, set B to 0
                        color[0] = uv[0]; // U -> Red
                        color[1] = uv[1]; // V -> Green
                        color[2] = 0;     // Blue = 0
                    }
                    uint32_t pixelIndex = (y * width + x) * 3;
                    frameBuffer[pixelIndex] = static_cast<uint8_t>(color[0] * 255);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>(color[1] * 255);
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>(color[2] * 255);
                });
            }

            else if (renderMode == RenderMode::DEPTH){
                std::vector<Ray<T>> rays;
                std::vector<T> depths(width * height);
                for (uint32_t i = 0; i < width * height; i++){
                    depths[i] = 0;
                }
                sampleNDC(invWorld2Cam, invProjectionMatrix, width, height, rays, 1, 1);
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    T depth = 0;
                    uint32_t i = y * width + x;
                    Intersection<T> intersection = trace(rays[i], primitives, bvhNodes);
                    if (intersection.hit){
                        depth = intersection.tHit;
                    }
                    depths[i] = depth;
                });

                T minDepth = MIN_DEPTH;
                T maxDepth = -MIN_DEPTH;
                for (uint32_t i = 0; i < width * height; i++){
                    minDepth = std::min(minDepth, depths[i]);
                    maxDepth = std::max(maxDepth, depths[i]);
                }

                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    uint32_t pixelIndex = (y * width + x) * 3;
                    T depth = depths[y * width + x];
                    frameBuffer[pixelIndex] = static_cast<uint8_t>((depth - minDepth) / (maxDepth - minDepth) * 255.0);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>((depth - minDepth) / (maxDepth - minDepth) * 255.0);
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>((depth - minDepth) / (maxDepth - minDepth) * 255.0);
                });
            }

            else if (renderMode == RenderMode::BASE_COLOR){
                // Render the base color of the hit point
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    std::vector<Ray<T>> rays;
                    sampleNDCThread(invWorld2Cam, invProjectionMatrix, width, height, rays, nPixelSamples, AAsubdivisions, x, y);
                    Eigen::Vector3<T> color = Eigen::Vector3<T>::Zero();
                    for (uint32_t j = 0; j < nPixelSamples; j++){
                        Intersection<T> intersection = trace(rays[j], primitives, bvhNodes);
                        if (intersection.hit){
                            Primitive<T>* primitive = primitives[intersection.primitiveID];
                            if (primitive->brdf != nullptr && primitive->brdf->pbrMap != nullptr){
                                color += primitive->brdf->pbrMap->baseColor(intersection.uv);
                            }
                        }
                    }
                    color /= nPixelSamples;
                    uint32_t pixelIndex = (y * width + x) * 3;
                    frameBuffer[pixelIndex] = static_cast<uint8_t>(color[0] * 255);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>(color[1] * 255);
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>(color[2] * 255);
                });
            }


            else if (renderMode == RenderMode::SHADING_NORMAL){
                // Render the shading normal of the hit point
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    std::vector<Ray<T>> rays;
                    sampleNDCThread(invWorld2Cam, invProjectionMatrix, width, height, rays, nPixelSamples, AAsubdivisions, x, y);
                    Eigen::Vector3<T> color = Eigen::Vector3<T>::Zero();
                    for (uint32_t j = 0; j < nPixelSamples; j++){
                        Intersection<T> intersection = trace(rays[j], primitives, bvhNodes);
                        if (intersection.hit){
                            Primitive<T>* primitive = primitives[intersection.primitiveID];
                            if (primitive->brdf != nullptr && primitive->brdf->pbrMap != nullptr){
                                color += transformShadingNormal(primitive->brdf->pbrMap->normal(intersection.uv), intersection.normal, intersection.tangent);
                            }
                        }
                    }
                    color /= nPixelSamples;
                    uint32_t pixelIndex = (y * width + x) * 3;
                    frameBuffer[pixelIndex] = static_cast<uint8_t>(color[0] * 255);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>(color[1] * 255);
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>(color[2] * 255);
                });
            }

            else if (renderMode == RenderMode::DIRECT_LIGHTING){
                // Render the direct lighting
                parallelFor(0, primitives.size(), [&](uint32_t i){
                    primitives[i]->projectAndRasterize(projectionMatrix, world2cam, frameBuffer, width, height);
                });
                std::cout << "<RayEngine> Rendering direct lighting" << std::endl;
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    std::vector<Ray<T>> rays;
                    sampleNDCThread(invWorld2Cam, invProjectionMatrix, width, height, rays, nPixelSamples, AAsubdivisions, x, y);
                    Eigen::Vector3<T> color = Eigen::Vector3<T>::Zero();
                    for (uint32_t j = 0; j < nPixelSamples; j++){
                        color += pathTracing(rays[j], primitives, bvhNodes, lights, hdriLight, nLightSamples, 1);
                    }
                    color /= nPixelSamples;
                    color = color.cwiseMax(Eigen::Vector3<T>::Zero());
                    color = color.cwiseMin(Eigen::Vector3<T>::Ones());
                    uint32_t pixelIndex = (y * width + x) * 3;
                    frameBuffer[pixelIndex] = static_cast<uint8_t>(color[0] * 255);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>(color[1] * 255);
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>(color[2] * 255);
                });
            }

            else if (renderMode == RenderMode::PATH_TRACING){
                // Render the path tracing
                parallelFor(0, primitives.size(), [&](uint32_t i){
                    primitives[i]->projectAndRasterize(projectionMatrix, world2cam, frameBuffer, width, height);
                });
                std::cout << "<RayEngine> Rendering path tracing" << std::endl;
                parallelFor2D(width, height, blockW, blockH, [&](uint32_t x, uint32_t y){
                    std::vector<Ray<T>> rays;
                    sampleNDCThread(invWorld2Cam, invProjectionMatrix, width, height, rays, nPixelSamples, AAsubdivisions, x, y);
                    Eigen::Vector3<T> color = Eigen::Vector3<T>::Zero();
                    for (uint32_t j = 0; j < nPixelSamples; j++){
                        color += pathTracing(rays[j], primitives, bvhNodes, lights, hdriLight, nLightSamples, maxPathDepth);
                    }
                    color /= nPixelSamples;
                    color = color.cwiseMax(Eigen::Vector3<T>::Zero());
                    color = color.cwiseMin(Eigen::Vector3<T>::Ones());
                    uint32_t pixelIndex = (y * width + x) * 3;
                    frameBuffer[pixelIndex] = static_cast<uint8_t>(color[0] * 255);
                    frameBuffer[pixelIndex + 1] = static_cast<uint8_t>(color[1] * 255);
                    frameBuffer[pixelIndex + 2] = static_cast<uint8_t>(color[2] * 255);
                });
            }


            isRendererRunning = false;
            isRendererFinished = true;
        }
    };
}

#endif
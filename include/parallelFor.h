#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include "utils.h"

namespace ray {
    /**
     * @brief Execute a function in parallel across a range of indices
     * 
     * This function has no auto-balancing, so it is recommended to use parallelFor2D for rendering
     * 
     * @param count The number of iterations to perform
     * @param func The lambda function to execute for each index
     */
    template<typename F>
    void parallelFor(uint32_t start, uint32_t end, F&& func) {

        // Get number of hardware threads available
        const uint32_t num_threads = std::thread::hardware_concurrency();
        
        // Calculate work per thread
        const uint32_t block_size = (end - start) / num_threads;
        const uint32_t remainder = (end - start) % num_threads;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        // Launch threads
        for (uint32_t i = 0; i < num_threads; ++i) {
            uint32_t end = start + block_size + (i < remainder ? 1 : 0);
            
            threads.emplace_back([start, end, &func]() {
                for (uint32_t j = start; j < end; ++j) {
                    func(j);
                }
            });

            start = end;
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }

    struct ParallelForBlock{
        uint32_t startX, endX, startY, endY;
    };

    /**
     * @brief Execute a function in parallel across a 2D grid of blocks
     * 
     * This function is designed to auto balance the workload across threads
     * 
     * @param width The width of the grid
     * @param height The height of the grid
     * @param blockW The width of each block
     * @param blockH The height of each block
     * @param func The lambda function to execute for each block f(x,y)
     */
    template<typename F>
    void parallelFor2D(uint32_t width, uint32_t height, uint32_t blockW, uint32_t blockH, F&& func){

        const uint32_t numBlocksX = (width + blockW - 1) / blockW;
        const uint32_t numBlocksY = (height + blockH - 1) / blockH;

        const uint32_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        // Calculate total number of blocks
        const uint32_t totalBlocks = numBlocksX * numBlocksY;
        
        // Create a queue of blocks to process
        std::queue<ParallelForBlock> blockQueue;
        std::mutex queueMutex;

        // Initialize queue with all blocks
        for (uint32_t blockY = 0; blockY < numBlocksY; blockY++) {
            for (uint32_t blockX = 0; blockX < numBlocksX; blockX++) {
                ParallelForBlock block;
                block.startX = blockX * blockW;
                block.endX = std::min(block.startX + blockW, width);
                block.startY = blockY * blockH;
                block.endY = std::min(block.startY + blockH, height);
                blockQueue.push(block);
            }
        }

        // Launch worker threads
        for (uint32_t i = 0; i < num_threads; ++i) {
            threads.emplace_back([&blockQueue, &queueMutex, &func, i]() {
                // Set thread affinity to specific CPU core
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(i, &cpuset);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

                
                while (true) {
                    // Get next block from queue
                    ParallelForBlock block;
                    {
                        std::lock_guard<std::mutex> lock(queueMutex);
                        if (blockQueue.empty()) {
                            break;
                        }
                        block = blockQueue.front();
                        blockQueue.pop();
                    }

                    // Process the block
                    for (uint32_t y = block.startY; y < block.endY; y++) {
                        for (uint32_t x = block.startX; x < block.endX; x++) {
                            func(x, y);
                        }
                    }
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    }
}

#endif
#pragma once

#include "AABB.h"
#include "LBVHPass.h"

#include <glm/glm.hpp>
#include <random>
#include <utility>

namespace engine {
    class LBVH {
    private:
        struct Element {
            uint32_t primitiveIdx;
            uint32_t paddingA;
            uint32_t paddingB;
            uint32_t paddingC;
            glm::ivec4 aabbMin;
            glm::ivec4 aabbMax;
        };

        struct MortonCodeElement {
            uint32_t mortonCode; // key for sorting
            uint32_t elementIdx; // pointer into element buffer
        };

#define INVALID_POINTER 0x0
#define ABSOLUTE_POINTERS 1
#define POINTER(index, pointer) (ABSOLUTE_POINTERS ? (pointer) : (index) + (pointer))

        struct LBVHNode {
            int32_t left;
            int32_t right;
            uint32_t primitiveIdx;
            uint32_t propertyIdx;
            glm::ivec4 aabbMin;
            glm::ivec4 aabbMax;
        };

        struct LBVHConstructionInfo {
            uint32_t parent;
            int32_t visitationCount;
        };

    public:
        void execute(GPUContext *gpuContext) {
            AABB extent{};
            std::vector<Element> elements;
            //            generateElements(elements, &extent);
            const uint NUM_ELEMENTS = elements.size();
            const uint NUM_LBVH_ELEMENTS = NUM_ELEMENTS + NUM_ELEMENTS - 1;
            bool absolutePointers = true;

            // gpu context
            m_gpuContext = gpuContext;

            // compute pass
            m_pass = std::make_shared<LBVHPass>(gpuContext);
            m_pass->create();
            m_pass->setGlobalInvocationSize(LBVHPass::MORTON_CODES, NUM_ELEMENTS, 1, 1);
            m_pass->setGlobalInvocationSize(LBVHPass::RADIX_SORT, 1, 1, 1);
            m_pass->setGlobalInvocationSize(LBVHPass::HIERARCHY, NUM_ELEMENTS, 1, 1);
            m_pass->setGlobalInvocationSize(LBVHPass::BOUNDING_BOXES, NUM_ELEMENTS, 1, 1);

            // push constants
            m_pass->m_pushConstantsMortonCodes.g_num_elements = NUM_ELEMENTS;
            m_pass->m_pushConstantsMortonCodes.g_min = glm::ivec3(0);
            m_pass->m_pushConstantsMortonCodes.g_max = glm::ivec3(400);
            m_pass->m_pushConstantsRadixSort.g_num_elements = NUM_ELEMENTS;
            m_pass->m_pushConstantsHierarchy.g_num_elements = NUM_ELEMENTS;
            m_pass->m_pushConstantsHierarchy.g_absolute_pointers = absolutePointers ? 1 : 0;
            m_pass->m_pushConstantsBoundingBoxes.g_num_elements = NUM_ELEMENTS;
            m_pass->m_pushConstantsBoundingBoxes.g_absolute_pointers = absolutePointers ? 1 : 0;

            // buffers
            auto settingsElement = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(NUM_ELEMENTS * sizeof(Element)), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "lbvh.elementsBuffer"};
            m_elementsBuffer = Buffer::fillDeviceWithStagingBuffer(m_gpuContext, settingsElement, elements.data());

            auto settingsMortonCode = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(NUM_ELEMENTS * sizeof(MortonCodeElement)), .m_bufferUsages = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "lbvh.mortonCodeBuffer"};
            m_mortonCodeBuffer = std::make_shared<Buffer>(gpuContext, settingsMortonCode);

            auto settingsMortonCodePingPong = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(NUM_ELEMENTS * sizeof(MortonCodeElement)), .m_bufferUsages = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "lbvh.mortonCodePingPongBuffer"};
            m_mortonCodePingPongBuffer = std::make_shared<Buffer>(gpuContext, settingsMortonCodePingPong);

            auto settingsLBVH = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(NUM_LBVH_ELEMENTS * sizeof(LBVHNode)), .m_bufferUsages = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "lbvh.LBVHBuffer"};
            m_LBVHBuffer = std::make_shared<Buffer>(gpuContext, settingsLBVH);

            auto settingsLBVHConstructionInfo = Buffer::BufferSettings{.m_sizeBytes = static_cast<uint32_t>(NUM_LBVH_ELEMENTS * sizeof(LBVHConstructionInfo)), .m_bufferUsages = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .m_memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, .m_name = "lbvh.LBVHConstructionInfoBuffer"};
            m_LBVHConstructionInfoBuffer = std::make_shared<Buffer>(gpuContext, settingsLBVHConstructionInfo);

            std::cout << PRINT_PREFIX << "Building LBVH for " << NUM_ELEMENTS << " elements." << std::endl;

            // set storage buffers
            m_pass->setStorageBuffer(0, 0, m_mortonCodeBuffer.get());
            m_pass->setStorageBuffer(0, 1, m_elementsBuffer.get());
            m_pass->setStorageBuffer(1, 0, m_mortonCodeBuffer.get());
            m_pass->setStorageBuffer(1, 1, m_mortonCodePingPongBuffer.get());
            m_pass->setStorageBuffer(2, 0, m_mortonCodeBuffer.get());
            m_pass->setStorageBuffer(2, 1, m_elementsBuffer.get());
            m_pass->setStorageBuffer(2, 2, m_LBVHBuffer.get());
            m_pass->setStorageBuffer(2, 3, m_LBVHConstructionInfoBuffer.get());
            m_pass->setStorageBuffer(3, 0, m_LBVHBuffer.get());
            m_pass->setStorageBuffer(3, 1, m_LBVHConstructionInfoBuffer.get());

            // execute pass
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            m_pass->execute(VK_NULL_HANDLE);
            vkQueueWaitIdle(m_gpuContext->m_queues->getQueue(Queues::COMPUTE));
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            double gpuTime = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) * std::pow(10, -3));
            std::cout << PRINT_PREFIX << "GPU build finished in " << gpuTime << "[ms]." << std::endl;

            // verify result
            verify(NUM_LBVH_ELEMENTS);

            // clean up
            releaseBuffers();
            m_pass->release();
        }

    private:
        GPUContext *m_gpuContext;

        std::shared_ptr<LBVHPass> m_pass;

        std::shared_ptr<Buffer> m_elementsBuffer;
        std::shared_ptr<Buffer> m_mortonCodeBuffer;
        std::shared_ptr<Buffer> m_mortonCodePingPongBuffer;
        std::shared_ptr<Buffer> m_LBVHBuffer;
        std::shared_ptr<Buffer> m_LBVHConstructionInfoBuffer;

        static inline const char *PRINT_PREFIX = "[LBVH] ";

        void releaseBuffers() {
            m_mortonCodeBuffer->release();
            m_mortonCodePingPongBuffer->release();
            m_elementsBuffer->release();
            m_LBVHBuffer->release();
            m_LBVHConstructionInfoBuffer->release();
        }

        void verify(uint numLBVHElements) {
            std::vector<LBVHNode> LBVH(numLBVHElements);
            m_LBVHBuffer->downloadWithStagingBuffer(LBVH.data());

            bool visited[LBVH.size()];
            traverse(0, LBVH.data(), visited);
            for (uint32_t i = 0; i < LBVH.size(); i++) {
                if (!visited[i]) {
                    std::cout << PRINT_PREFIX << "Error: Node not visited." << std::endl;
                    throw std::runtime_error("TEST FAILED.");
                }
            }

            // verify aabbs

            std::cout << PRINT_PREFIX << "Verification successful." << std::endl;
        }

        void traverse(uint32_t index, LBVHNode *LBVH, bool *visited) {
            LBVHNode node = LBVH[index];

            if (node.left == INVALID_POINTER && node.right != INVALID_POINTER
                || node.left != INVALID_POINTER && node.right == INVALID_POINTER) {
                std::cout << PRINT_PREFIX << "Error: Node has only one child." << std::endl;
                throw std::runtime_error("TEST FAILED.");
            }

            if (node.left == INVALID_POINTER) {
                // leaf
                if (visited[index]) {
                    std::cout << PRINT_PREFIX << "Error: Leaf node visited twice." << std::endl;
                    throw std::runtime_error("TEST FAILED.");
                }
                visited[index] = true;
            } else {
                // inner node
                if (visited[index]) {
                    std::cout << PRINT_PREFIX << "Error: Inner node visited twice." << std::endl;
                    throw std::runtime_error("TEST FAILED.");
                }
                visited[index] = true;

                uint32_t leftChildIndex = POINTER(index, node.left);
                uint32_t rightChildIndex = POINTER(index, node.right);
                traverse(leftChildIndex, LBVH, visited);
                traverse(rightChildIndex, LBVH, visited);
            }
        }
    };
} // namespace engine
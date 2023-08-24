#include "LBVH.h"

namespace engine {

    void LBVH::execute(GPUContext *gpuContext) {
        AABB extent{};
        std::vector<Element> elements;
        generateElements(elements, &extent);
        const uint NUM_ELEMENTS = elements.size();
        const uint NUM_LBVH_ELEMENTS = NUM_ELEMENTS + NUM_ELEMENTS - 1;

        // gpu context
        m_gpuContext = gpuContext;

        // compute pass
        m_pass = std::make_shared<LBVHPass>(gpuContext);
        m_pass->create();
        m_pass->setGlobalInvocationSize(LBVHPass::MORTON_CODES, NUM_ELEMENTS, 1, 1);
        m_pass->setGlobalInvocationSize(LBVHPass::RADIX_SORT, 256, 1, 1); // WORKGROUP_SIZE defined in lbvh_single_radix_sort.comp, i.e. we just want to launch a single work group
        m_pass->setGlobalInvocationSize(LBVHPass::HIERARCHY, NUM_ELEMENTS, 1, 1);
        m_pass->setGlobalInvocationSize(LBVHPass::BOUNDING_BOXES, NUM_ELEMENTS, 1, 1);

        // push constants
        m_pass->m_pushConstantsMortonCodes.g_num_elements = NUM_ELEMENTS;
        m_pass->m_pushConstantsMortonCodes.g_min_x = 8 * extent.min.x;
        m_pass->m_pushConstantsMortonCodes.g_min_y = 8 * extent.min.y;
        m_pass->m_pushConstantsMortonCodes.g_min_z = 8 * extent.min.z;
        m_pass->m_pushConstantsMortonCodes.g_max_x = 8 * extent.max.x;
        m_pass->m_pushConstantsMortonCodes.g_max_y = 8 * extent.max.y;
        m_pass->m_pushConstantsMortonCodes.g_max_z = 8 * extent.max.z;
        m_pass->m_pushConstantsRadixSort.g_num_elements = NUM_ELEMENTS;
        m_pass->m_pushConstantsHierarchy.g_num_elements = NUM_ELEMENTS;
        m_pass->m_pushConstantsHierarchy.g_absolute_pointers = ABSOLUTE_POINTERS;
        m_pass->m_pushConstantsBoundingBoxes.g_num_elements = NUM_ELEMENTS;
        m_pass->m_pushConstantsBoundingBoxes.g_absolute_pointers = ABSOLUTE_POINTERS;

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
        std::cout << PRINT_PREFIX << "Union of all element AABBs: " << extent << std::endl;

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

    void LBVH::releaseBuffers() {
        m_mortonCodeBuffer->release();
        m_mortonCodePingPongBuffer->release();
        m_elementsBuffer->release();
        m_LBVHBuffer->release();
        m_LBVHConstructionInfoBuffer->release();
    }

    void LBVH::verify(uint numLBVHElements) {
        std::vector<LBVHNode> LBVH(numLBVHElements);
        m_LBVHBuffer->downloadWithStagingBuffer(LBVH.data());

        std::cout << PRINT_PREFIX << "Writing LBVH to file (lbvh.csv)..." << std::endl;

        std::ofstream myfile;
        myfile.open("lbvh.csv");
        myfile << "left right primitiveIdx aabb_min_x aabb_min_y aabb_min_z aabb_max_x aabb_max_y aabb_max_z\n";
        for (uint32_t i = 0; i < numLBVHElements; i++) {
            myfile << LBVH[i].left << " "
                   << LBVH[i].right << " "
                   << LBVH[i].primitiveIdx << " "
                   << LBVH[i].aabbMinX << " "
                   << LBVH[i].aabbMinY << " "
                   << LBVH[i].aabbMinZ << " "
                   << LBVH[i].aabbMaxX << " "
                   << LBVH[i].aabbMaxY << " "
                   << LBVH[i].aabbMaxZ << "\n";
        }
        myfile.close();

        std::cout << PRINT_PREFIX << "Writing successful." << std::endl;

        std::cout << PRINT_PREFIX << "Starting verification of hierarchy and bounding boxes..." << std::endl;

        std::vector<bool> visited(numLBVHElements, false);
        traverse(0, LBVH.data(), visited);
        for (uint32_t i = 0; i < LBVH.size(); i++) {
            if (!visited[i]) {
                std::cout << PRINT_PREFIX << "Error: Node not visited." << std::endl;
                throw std::runtime_error("TEST FAILED.");
            }
        }

        std::cout << PRINT_PREFIX << "Verification successful." << std::endl;
    }

    bool LBVH::aabbIsUnion(AABB parentAABB, AABB childAAABB, AABB childBAABB) {
        AABB childrenAABB;
        childrenAABB.expand(childAAABB.min);
        childrenAABB.expand(childAAABB.max);
        childrenAABB.expand(childBAABB.min);
        childrenAABB.expand(childBAABB.max);
        float EPS = 0.0001;
        if (glm::abs(parentAABB.max.x - childrenAABB.max.x) > EPS ||
            glm::abs(parentAABB.max.y - childrenAABB.max.y) > EPS ||
            glm::abs(parentAABB.max.z - childrenAABB.max.z) > EPS ||
            glm::abs(parentAABB.min.x - childrenAABB.min.x) > EPS ||
            glm::abs(parentAABB.min.y - childrenAABB.min.y) > EPS ||
            glm::abs(parentAABB.min.z - childrenAABB.min.z) > EPS) {
            return false;
        }
        return true;
    }

    void LBVH::traverse(uint32_t index, LBVH::LBVHNode *LBVH, std::vector<bool> &visited) {
        LBVHNode node = LBVH[index];

        if (node.left == INVALID_POINTER && node.right != INVALID_POINTER || node.left != INVALID_POINTER && node.right == INVALID_POINTER) {
            std::cout << PRINT_PREFIX << "Error: Node " << index << " has only one child." << std::endl;
            throw std::runtime_error("TEST FAILED.");
        }

        if (node.left == INVALID_POINTER) {
            // leaf
            if (visited[index]) {
                std::cout << PRINT_PREFIX << "Error: Leaf node " << index << " visited twice." << std::endl;
                throw std::runtime_error("TEST FAILED.");
            }
            visited[index] = true;
        } else {
            // inner node
            if (visited[index]) {
                std::cout << PRINT_PREFIX << "Error: Inner node " << index << " visited twice." << std::endl;
                throw std::runtime_error("TEST FAILED.");
            }
            visited[index] = true;

            uint32_t leftChildIndex = POINTER(index, node.left);
            uint32_t rightChildIndex = POINTER(index, node.right);

            // verify aabbs
            LBVHNode childA = LBVH[leftChildIndex];
            LBVHNode childB = LBVH[rightChildIndex];
            AABB parentAABB({node.aabbMinX, node.aabbMinY, node.aabbMinZ, 0}, {node.aabbMaxX, node.aabbMaxY, node.aabbMaxZ, 0});
            AABB childAAABB({childA.aabbMinX, childA.aabbMinY, childA.aabbMinZ, 0}, {childA.aabbMaxX, childA.aabbMaxY, childA.aabbMaxZ, 0});
            AABB childBAABB({childB.aabbMinX, childB.aabbMinY, childB.aabbMinZ, 0}, {childB.aabbMaxX, childB.aabbMaxY, childB.aabbMaxZ, 0});
            if (!aabbIsUnion(parentAABB, childAAABB, childBAABB)) {
                std::cout << PRINT_PREFIX << "Error: Inner node " << index << " has an AABB that is not the union of the children (left=" << leftChildIndex << ",right=" << rightChildIndex << ") AABBs. parentAABB=" << parentAABB << " leftAABB=" << childAAABB << " rightAABB=" << childBAABB << std::endl;
                throw std::runtime_error("TEST FAILED.");
            }

            // continue traversal
            traverse(leftChildIndex, LBVH, visited);
            traverse(rightChildIndex, LBVH, visited);
        }
    }

    void LBVH::generateElements(std::vector<Element> &elements, AABB *extent) {
        const std::string MODEL_FILE_NAME = "dragon.obj";
        const std::string MODEL_PATH_DIRECTORY = engine::Paths::m_resourceDirectoryPath + "/models";

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, (MODEL_PATH_DIRECTORY + "/" + MODEL_FILE_NAME).c_str(), MODEL_PATH_DIRECTORY.c_str())) {
            throw std::runtime_error(warn + err);
        }

        uint32_t primitiveIndex = 0;
        // Loop over shapes
        for (auto &shape: shapes) {
            // Loop over faces(polygon)
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                auto fv = size_t(shape.mesh.num_face_vertices[f]);
                if (fv != 3) {
                    std::cout << PRINT_PREFIX << "Error: Only triangle meshes supported." << std::endl;
                    throw std::runtime_error("TEST FAILED.");
                }

                auto minV = glm::vec3(std::numeric_limits<float>::max());
                auto maxV = glm::vec3(-std::numeric_limits<float>::max());

                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; v++) {
                    // access to vertex
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                    minV[0] = glm::min(minV[0], attrib.vertices[3 * size_t(idx.vertex_index) + 0]);
                    minV[1] = glm::min(minV[1], attrib.vertices[3 * size_t(idx.vertex_index) + 1]);
                    minV[2] = glm::min(minV[2], attrib.vertices[3 * size_t(idx.vertex_index) + 2]);

                    maxV[0] = glm::max(maxV[0], attrib.vertices[3 * size_t(idx.vertex_index) + 0]);
                    maxV[1] = glm::max(maxV[1], attrib.vertices[3 * size_t(idx.vertex_index) + 1]);
                    maxV[2] = glm::max(maxV[2], attrib.vertices[3 * size_t(idx.vertex_index) + 2]);
                }
                index_offset += fv;

                AABB aabb;
                aabb.expand(minV);
                aabb.expand(maxV);
                elements.push_back({primitiveIndex, aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z});
                primitiveIndex++;
                extent->expand(minV);
                extent->expand(maxV);
            }
        }
    }
}
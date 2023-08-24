#pragma once

#include "AABB.h"
#include "LBVHPass.h"
#include "tinyobjloader/tiny_obj_loader.h"

#include <glm/glm.hpp>
#include <random>
#include <utility>

namespace engine {
    class LBVH {
    private:
        // input for the builder (normally a triangle or some other kind of primitive); it is necessary to allocate and fill the buffer
        struct Element {
            uint32_t primitiveIdx; // the id of the primitive; this primitive id is copied to the leaf nodes of the  LBVHNode
            float aabbMinX;        // aabb of the primitive
            float aabbMinY;
            float aabbMinZ;
            float aabbMaxX;
            float aabbMaxY;
            float aabbMaxZ;
        };

        // output of the builder; it is necessary to allocate the (empty) buffer
        struct LBVHNode {
            int32_t left;          // pointer to the left child or INVALID_POINTER in case of leaf
            int32_t right;         // pointer to the right child or INVALID_POINTER in case of leaf
            uint32_t primitiveIdx; // custom value that is copied from the input Element or 0 in case of inner node
            float aabbMinX;        // aabb of the node
            float aabbMinY;
            float aabbMinZ;
            float aabbMaxX;
            float aabbMaxY;
            float aabbMaxZ;
        };

        // only used on the GPU side during construction; it is necessary to allocate the (empty) buffer
        struct MortonCodeElement {
            uint32_t mortonCode; // key for sorting
            uint32_t elementIdx; // pointer into element buffer
        };

        // only used on the GPU side during construction; it is necessary to allocate the (empty) buffer
        struct LBVHConstructionInfo {
            uint32_t parent;         // pointer to the parent
            int32_t visitationCount; // number of threads that arrived
        };

#define INVALID_POINTER 0x0 // do not change
#define ABSOLUTE_POINTERS 1 // 1 to use absolute pointers (left/right child pointer is the absolute index of the child in the buffer/array)
//or 0 for relative pointers (left/right child pointer is the relative pointer from the parent index to the child index in the buffer, i.e. absolute child pointer = absolute parent pointer + relative child pointer)
#define POINTER(index, pointer) (ABSOLUTE_POINTERS ? (pointer) : (index) + (pointer)) // helper macro to handle relative pointers on CPU side, i.e. convert them to absolute pointers for array indexing

    public:
        void execute(GPUContext *gpuContext);

    private:
        GPUContext *m_gpuContext;

        std::shared_ptr<LBVHPass> m_pass;

        std::shared_ptr<Buffer> m_elementsBuffer;
        std::shared_ptr<Buffer> m_mortonCodeBuffer;
        std::shared_ptr<Buffer> m_mortonCodePingPongBuffer;
        std::shared_ptr<Buffer> m_LBVHBuffer;
        std::shared_ptr<Buffer> m_LBVHConstructionInfoBuffer;

        static inline const char *PRINT_PREFIX = "[LBVH] ";

        void releaseBuffers();

        void verify(uint numLBVHElements);

        static bool aabbIsUnion(AABB parentAABB, AABB childAAABB, AABB childBAABB);

        void traverse(uint32_t index, LBVHNode *LBVH, std::vector<bool> &visited);

        static void generateElements(std::vector<Element> &elements, AABB *extent);
    };
} // namespace engine
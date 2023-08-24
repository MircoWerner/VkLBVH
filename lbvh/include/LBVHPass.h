#pragma once

#include "engine/util/Paths.h"
#include "engine/passes/ComputePass.h"

namespace engine {
    class LBVHPass : public ComputePass {
    public:
        explicit LBVHPass(GPUContext *gpuContext) : ComputePass(gpuContext) {
        }

        enum ComputeStage {
            MORTON_CODES = 0,
            RADIX_SORT = 1,
            HIERARCHY = 2,
            BOUNDING_BOXES = 3,
        };

        struct PushConstantsMortonCodes {
            uint32_t g_num_elements;
            float g_min_x;
            float g_min_y;
            float g_min_z;
            float g_max_x;
            float g_max_y;
            float g_max_z;
        };
        PushConstantsMortonCodes m_pushConstantsMortonCodes{};

        struct PushConstantsRadixSort {
            uint32_t g_num_elements;
        };
        PushConstantsRadixSort m_pushConstantsRadixSort{};

        struct PushConstantsHierarchy {
            uint32_t g_num_elements;
            uint32_t g_absolute_pointers;
        };
        PushConstantsHierarchy m_pushConstantsHierarchy{};

        struct PushConstantsBoundingBoxes {
            uint32_t g_num_elements;
            uint32_t g_absolute_pointers;
        };
        PushConstantsBoundingBoxes m_pushConstantsBoundingBoxes{};

    protected:
        std::vector<std::shared_ptr<Shader>> createShaders() override;

        void recordCommands(VkCommandBuffer commandBuffer) override;

        void createPipelineLayouts() override;
    };
}
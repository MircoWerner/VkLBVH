#include "LBVHPass.h"

namespace engine {

    std::vector<std::shared_ptr<Shader>> LBVHPass::createShaders() {
        return {std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "lbvh_morton_codes.comp"),
                std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "lbvh_single_radixsort.comp"),
                std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "lbvh_hierarchy.comp"),
                std::make_shared<Shader>(m_gpuContext, Paths::m_resourceDirectoryPath + "/shaders", "lbvh_bounding_boxes.comp")};
    }

    void LBVHPass::recordCommands(VkCommandBuffer commandBuffer) {
        vkCmdPushConstants(commandBuffer, m_pipelineLayouts[MORTON_CODES], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsMortonCodes), &m_pushConstantsMortonCodes);
        recordCommandComputeShaderExecution(commandBuffer, MORTON_CODES);
        VkMemoryBarrier memoryBarrier0{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier0, 0, nullptr, 0, nullptr);

        vkCmdPushConstants(commandBuffer, m_pipelineLayouts[RADIX_SORT], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsRadixSort), &m_pushConstantsRadixSort);
        recordCommandComputeShaderExecution(commandBuffer, RADIX_SORT);
        VkMemoryBarrier memoryBarrier1{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier1, 0, nullptr, 0, nullptr);

        vkCmdPushConstants(commandBuffer, m_pipelineLayouts[HIERARCHY], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsHierarchy), &m_pushConstantsHierarchy);
        recordCommandComputeShaderExecution(commandBuffer, HIERARCHY);
        VkMemoryBarrier memoryBarrier2{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier2, 0, nullptr, 0, nullptr);

        vkCmdPushConstants(commandBuffer, m_pipelineLayouts[BOUNDING_BOXES], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantsBoundingBoxes), &m_pushConstantsBoundingBoxes);
        recordCommandComputeShaderExecution(commandBuffer, BOUNDING_BOXES);
        VkMemoryBarrier memoryBarrier3{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, {}, 1, &memoryBarrier3, 0, nullptr, 0, nullptr);
    }

    void LBVHPass::createPipelineLayouts() {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = m_descriptorSetLayouts.size();
        pipelineLayoutInfo.pSetLayouts = m_descriptorSetLayouts.data();

        // BOUNDING_BOXES
        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstantsMortonCodes);

        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(m_gpuContext->m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayouts[MORTON_CODES]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // RADIX_SORT
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstantsRadixSort);

        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(m_gpuContext->m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayouts[RADIX_SORT]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // HIERARCHY
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstantsHierarchy);

        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(m_gpuContext->m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayouts[HIERARCHY]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }

        // RADIX_SORT
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstantsBoundingBoxes);

        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        if (vkCreatePipelineLayout(m_gpuContext->m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayouts[BOUNDING_BOXES]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create pipeline layout!");
        }
    }
} // namespace engine
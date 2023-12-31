#pragma once

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "SPIRV-Reflect/spirv_reflect.h"
#include "Uniform.h"

namespace engine {
    class Shader {
    public:
        struct DescriptorSetLayoutData {
            uint32_t set_number{};
            VkDescriptorSetLayoutCreateInfo create_info{};
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            std::map<uint32_t, uint32_t> bindingToIndex;

            void generateBindingToIndexMap() {
                for (uint32_t i = 0; i < bindings.size(); i++) {
                    const auto &binding = bindings[i];
                    bindingToIndex[binding.binding] = i;
                }
            }
        };

        Shader(GPUContext *gpuContext, const std::string &inputPath, const std::string &fileName) : m_gpuContext(gpuContext) {
            std::cout << "[Shader] Compiling " << inputPath << "/" << fileName << std::endl;

            std::stringstream outputPath;
            outputPath << std::filesystem::canonical("/proc/self/exe").remove_filename().c_str() << "resources/shaders";
            compileShader(inputPath, outputPath.str(), fileName);
            std::vector<char> code = readFile(outputPath.str() + "/" + fileName + ".spv");

            reflect(code);

            m_shaderModule = createShaderModule(code, m_gpuContext->m_device);
        }

        ~Shader() {
            release();
        }

        void release() {
            if (m_shaderModule) {
                vkDestroyShaderModule(m_gpuContext->m_device, m_shaderModule, nullptr);
            }
            m_shaderModule = nullptr;
        }

        VkPipelineShaderStageCreateInfo generateShaderStageCreateInfo(VkShaderStageFlagBits shaderStage) {
            VkPipelineShaderStageCreateInfo shaderStageInfo{};
            shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            shaderStageInfo.stage = shaderStage;
            shaderStageInfo.module = m_shaderModule;
            shaderStageInfo.pName = "main";
            return shaderStageInfo;
        }

        std::map<uint32_t, DescriptorSetLayoutData> *getDescriptorSetLayoutData() {
            return &m_descriptorSetLayoutData;
        }

        std::map<uint32_t, std::vector<std::shared_ptr<Uniform>>> *getUniforms() {
            return &m_uniforms;
        }

        [[nodiscard]] VkExtent3D getWorkGroupSize() const {
            return m_workGroupSize;
        }

    private:
        GPUContext *m_gpuContext;
        VkShaderModule m_shaderModule;

        std::map<uint32_t, DescriptorSetLayoutData> m_descriptorSetLayoutData;
        std::map<uint32_t, std::vector<std::shared_ptr<Uniform>>> m_uniforms;

        VkExtent3D m_workGroupSize = {0, 0, 0};

        static VkShaderModule createShaderModule(const std::vector<char> &code, VkDevice device) {
            VkShaderModuleCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

            VkShaderModule shaderModule;
            if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create shader module!");
            }

            return shaderModule;
        }

        static std::vector<char> readFile(const std::string &filename) {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file!");
            }
            size_t fileSize = (size_t) file.tellg();
            std::vector<char> buffer(fileSize);
            file.seekg(0);
            file.read(buffer.data(), static_cast<std::streamsize>(fileSize));
            file.close();
            return buffer;
        }

        static void compileShader(const std::string &inputPath, const std::string &outputPath, const std::string &fileName) {
            std::filesystem::create_directories(outputPath);

            std::stringstream cmd;
            cmd << "glslc --target-spv=spv1.5 " << inputPath << "/" << fileName << " -o " << outputPath << "/" << fileName << ".spv";

            std::string cmd_output;
            char read_buffer[1024];
            FILE *cmd_stream = popen(cmd.str().c_str(), "r");
            while (fgets(read_buffer, sizeof(read_buffer), cmd_stream))
                cmd_output += read_buffer;
            int cmd_ret = pclose(cmd_stream);

            if (cmd_ret != 0) {
                throw std::runtime_error("unable to compile");
            }
        }

        void reflect(const std::vector<char> &code) {
            SpvReflectShaderModule module = {};
            SpvReflectResult result =
                    spvReflectCreateShaderModule(sizeof(code[0]) * code.size(), code.data(), &module);
            assert(result == SPV_REFLECT_RESULT_SUCCESS);

            reflectDescriptorSetLayout(module);
            reflectWorkGroupSize(module);

            spvReflectDestroyShaderModule(&module);
        }

        void reflectDescriptorSetLayout(const SpvReflectShaderModule &module) {
            uint32_t count = 0;
            SpvReflectResult result = spvReflectEnumerateDescriptorSets(&module, &count, nullptr);
            assert(result == SPV_REFLECT_RESULT_SUCCESS);

            std::vector<SpvReflectDescriptorSet *> sets(count);
            result = spvReflectEnumerateDescriptorSets(&module, &count, sets.data());
            assert(result == SPV_REFLECT_RESULT_SUCCESS);

            for (const auto &reflectedSet: sets) {
                m_descriptorSetLayoutData[reflectedSet->set] = {};
                DescriptorSetLayoutData &layout = m_descriptorSetLayoutData[reflectedSet->set];

                layout.bindings.resize(reflectedSet->binding_count);
                for (uint32_t i_binding = 0; i_binding < reflectedSet->binding_count; ++i_binding) {
                    const SpvReflectDescriptorBinding &refl_binding = *(reflectedSet->bindings[i_binding]);
                    VkDescriptorSetLayoutBinding &layout_binding = layout.bindings[i_binding];
                    layout_binding.binding = refl_binding.binding;
                    layout_binding.descriptorType = static_cast<VkDescriptorType>(refl_binding.descriptor_type);
                    layout_binding.descriptorCount = 1;
                    for (uint32_t i_dim = 0; i_dim < refl_binding.array.dims_count; ++i_dim) {
                        layout_binding.descriptorCount *= refl_binding.array.dims[i_dim];
                    }
                    layout_binding.stageFlags = static_cast<VkShaderStageFlagBits>(module.shader_stage);

                    if (layout_binding.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
                        // reflect uniform
                        m_uniforms[reflectedSet->set].push_back(Uniform::reflect(m_gpuContext, refl_binding));
                    }
                }
                layout.set_number = reflectedSet->set;
                layout.create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                layout.create_info.bindingCount = reflectedSet->binding_count;
                layout.create_info.pBindings = layout.bindings.data();
            }
        }

        void reflectWorkGroupSize(const SpvReflectShaderModule &module) {
            auto entryPoint = spvReflectGetEntryPoint(&module, "main");
            if (entryPoint != nullptr) {
                m_workGroupSize = {entryPoint->local_size.x, entryPoint->local_size.y, entryPoint->local_size.z};
            }
        }
    };
} // namespace engine
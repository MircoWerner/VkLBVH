#pragma once

#include <glm/glm.hpp>
#include <iostream>

namespace engine {
    struct AABB {
        glm::vec4 min = glm::vec4(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0);
        glm::vec4 max = glm::vec4(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), 0);

        void expand(glm::vec3 vec) {
            if (min.x > vec.x) {
                min.x = vec.x;
            }
            if (min.y > vec.y) {
                min.y = vec.y;
            }
            if (min.z > vec.z) {
                min.z = vec.z;
            }

            if (max.x < vec.x) {
                max.x = vec.x;
            }
            if (max.y < vec.y) {
                max.y = vec.y;
            }
            if (max.z < vec.z) {
                max.z = vec.z;
            }
        }

        [[nodiscard]] float calculateVolume() const {
            if (min.x >= max.x || min.y >= max.y || min.z >= max.z) {
                return 0;
            }
            return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
        }

        [[nodiscard]] float maxExtent() const { return glm::max(0.f, glm::max(max.x - min.x, glm::max(max.y - min.y, max.z - min.z))) + 1.f; };

        [[nodiscard]] int maxExtentAxis() const {
            float xExtent = max.x - min.x;
            float yExtent = max.y - min.y;
            float zExtent = max.z - min.z;
            if (xExtent > yExtent && xExtent > zExtent) {
                return 0;
            }
            return yExtent > zExtent ? 1 : 2;
        }

        [[nodiscard]] float maxElement() const { return glm::max(max.x, glm::max(max.y, max.z)); }

        [[nodiscard]] float minElement() const { return glm::min(min.x, glm::min(min.y, min.z)); }

        friend std::ostream& operator<< (std::ostream& stream, const AABB& aabb) {
            stream << "AABB{ min=(" << aabb.min.x << "," << aabb.min.y << "," << aabb.min.z << "), max=(" << aabb.max.x << "," << aabb.max.y << "," << aabb.max.z << ") }";
            return stream;
        }
    };
} // namespace engine
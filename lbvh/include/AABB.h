#pragma once

#include <glm/glm.hpp>
#include <iostream>

namespace engine {
    struct AABB {
        glm::ivec4 min = glm::ivec4(INT32_MAX, INT32_MAX, INT32_MAX, 0);
        glm::ivec4 max = glm::ivec4(INT32_MIN, INT32_MIN, INT32_MIN, 0);

        void expand(glm::ivec3 vec) {
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

        [[nodiscard]] uint32_t calculateVolume() const {
            if (min.x >= max.x || min.y >= max.y || min.z >= max.z) {
                return 0;
            }
            return (max.x - min.x) * (max.y - min.y) * (max.z - min.z);
        }

        [[nodiscard]] uint32_t maxExtent() const { return static_cast<uint32_t>(glm::max(0, glm::max(max.x - min.x, glm::max(max.y - min.y, max.z - min.z)))) + 1; };

        [[nodiscard]] int maxExtentAxis() const {
            int xExtent = max.x - min.x;
            int yExtent = max.y - min.y;
            int zExtent = max.z - min.z;
            if (xExtent > yExtent && xExtent > zExtent) {
                return 0;
            }
            return yExtent > zExtent ? 1 : 2;
        }

        [[nodiscard]] int32_t maxElement() const { return glm::max(max.x, glm::max(max.y, max.z)); }

        [[nodiscard]] int32_t minElement() const { return glm::min(min.x, glm::min(min.y, min.z)); }

        friend std::ostream& operator<< (std::ostream& stream, const AABB& aabb) {
            stream << "AABB{ min=(" << aabb.min.x << "," << aabb.min.y << "," << aabb.min.z << "), max=(" << aabb.max.x << "," << aabb.max.y << "," << aabb.max.z << ") }";
            return stream;
        }
    };
} // namespace engine
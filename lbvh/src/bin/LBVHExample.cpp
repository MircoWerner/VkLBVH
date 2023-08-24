#define TINYOBJLOADER_IMPLEMENTATION

#include "LBVH.h"
#include "engine/core/GPUContext.h"
#include "engine/util/Paths.h"

int main() {
#ifdef RESOURCE_DIRECTORY_PATH
    engine::Paths::m_resourceDirectoryPath = RESOURCE_DIRECTORY_PATH;
#endif

    engine::GPUContext gpu(engine::Queues::QueueFamilies::COMPUTE_FAMILY | engine::Queues::TRANSFER_FAMILY);

    try {
        gpu.init();

        auto app = std::make_shared<engine::LBVH>();
        app->execute(&gpu);

        gpu.shutdown();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
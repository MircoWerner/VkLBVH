cmake_minimum_required(VERSION 3.18)
project(lbvh VERSION 0.1.0 DESCRIPTION "Vulkan Linear BVH" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(lib)
add_subdirectory(engine)

option(MAKE_LBVH_EXAMPLE "Build Vulkan LBVH Example." ON)
if (MAKE_LBVH_EXAMPLE)
	add_subdirectory(lbvh)
endif()

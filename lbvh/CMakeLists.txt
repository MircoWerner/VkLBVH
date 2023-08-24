cmake_minimum_required(VERSION 3.18)
project(lbvhexample VERSION 0.1.0 DESCRIPTION "Vulkan LBVH Example" LANGUAGES CXX)

set(PROJECT_HEADERS
        include/LBVH.h
        include/LBVHPass.h
        include/AABB.h)

set(PROJECT_SOURCES
        src/bin/LBVHExample.cpp
        src/LBVH.cpp
        src/LBVHPass.cpp
)

add_executable(lbvhexample ${PROJECT_HEADERS} ${PROJECT_SOURCES})

target_link_libraries(lbvhexample Vulkan::Vulkan enginecore spirv-reflect tinyobjloader)

target_include_directories(lbvhexample
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        )

SET(RESOURCE_DIRECTORY_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/resources\")
if (RESOURCE_DIRECTORY_PATH)
    target_compile_definitions(lbvhexample PRIVATE RESOURCE_DIRECTORY_PATH=${RESOURCE_DIRECTORY_PATH})
endif()

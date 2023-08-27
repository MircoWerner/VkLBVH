# Vulkan LBVH (Linear Bounding Volume Hierarchy)

**GPU LBVH builder** implemented in **Vulkan** and **GLSL**.

The implementation is based on the following paper:
- [Tero Karras. 2012. Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. In Eurographics/ ACM SIGGRAPH Symposium on High Performance Graphics, The Eurographics Association. DOI:https://doi.org/10.2312/EGGH/HPG12/033-037](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)

and inspired by / based on the following repositories and blog posts:
- [LBVH implementation: LBVH in CUDA by ToruNiina](https://github.com/ToruNiina/lbvh)
- [LBVH blog post: Tree Construction on the GPU by Tero Karras](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)

and uses the `single_radixsort` from my [VkRadixSort](https://github.com/MircoWerner/VkRadixSort) repository.

Tested on Linux with NVIDIA RTX 3070 GPU and on Linux with AMD Radeon RX Vega 7 GPU.

## (IMPORTANT) NVIDIA vs. AMD
The shaders are configured for NVIDIA GPUs. If you are using an AMD GPU, set `SUBGROUP_SIZE=64` in [lbvh_single_radixsort.comp](https://github.com/MircoWerner/VkLBVH/blob/main/lbvh/resources/shaders/lbvh_single_radixsort.comp#L15).

## Table of Contents
- [Example Usage](#example-usage) (reference implementation in Vulkan)
  - [Compile / Run](#compile--run)
  - [Interesting Files](#interesting-files)
- [Own Usage](#own-usage) (how to use the LBVH builder / the compute shaders in your own Vulkan project)
  - [Struct Definition](#struct-definition)
  - [Model Loading](#model-loading)
  - [Shaders / Compute Pass](#shaders--compute-pass)
  - [Buffers](#buffers)
  - [Execute](#execute-)
- [Screenshot](#screenshot) (visualization of the constructed LBVH)

<a name="example--usage"></a>
## Example Usage
This repository contains a reference implementation to show how to use the provided shaders for the LBVH builder.

<a name="compile--run"></a>
### Compile / Run
Requirements: Vulkan, glm

```bash 
git clone --recursive git@github.com:MircoWerner/VkLBVH.git
cd VkLBVH
mkdir build
cd build
cmake ..
make
cd lbvh
./lbvhexample
```

<a name="interesting--files"></a>
### Interesting Files
- LBVH builder shaders `lbvh/resources/shaders`
- LBVH compute pass `lbvh/include/LBVHPass.h` `lbvh/src/LBVHPass.cpp`
- Program logic (buffer definition, assigning push constants, execution...) `lbvh/include/LBVH.h` `lbvh/src/LBVH.cpp`

<a name="own--usage"></a>
## Own Usage
Explanation how to use the LBVH builder in your own Vulkan project.

<a name="struct--definition"></a>
### Struct Definition
Define the following structs:
```cpp
#define INVALID_POINTER 0x0

// input for the builder (normally a triangle or some other kind of primitive); it is necessary to allocate the buffer on the GPU
// and to upload the input data
struct Element {
    uint32_t primitiveIdx; // the id of the primitive; this primitive id is copied to the leaf nodes of the BVH (LBVHNode)
    float aabbMinX;        // aabb of the primitive
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

// output of the builder; it is necessary to allocate the (empty) buffer on the GPU
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

// only used on the GPU side during construction; it is necessary to allocate the (empty) buffer on the GPU
struct MortonCodeElement {
    uint32_t mortonCode; // key for sorting
    uint32_t elementIdx; // pointer into element buffer
};

// only used on the GPU side during construction; it is necessary to allocate the (empty) buffer on the GPU
struct LBVHConstructionInfo {
    uint32_t parent;         // pointer to the parent
    int32_t visitationCount; // number of threads that arrived
};
```

<a name="model--loading"></a>
### Model Loading
Load some model containing `NUM_ELEMENTS` primitives. The LBVH will contain `NUM_LBVH_ELEMENTS = NUM_ELEMENTS + NUM_ELEMENTS - 1;` nodes after building.
Define a vector/array containing `Element` structs for all primitives. This vector/array will be uploaded to the input (element) buffer for building.

<a name="shaders--compute-pass"></a>
### Shaders / Compute Pass
Copy the following [shaders](https://github.com/MircoWerner/VkLBVH/tree/main/lbvh/resources/shaders) to your project:
```
lbvh_morton_codes.comp: assign morton codes to the input elements
lbvh_single_radixsort.comp: sort the morton codes
lbvh_hierarchy.comp: build the bvh hierarchy
lbvh_bounding_boxes.comp: build the aabbs

lbvh_common.glsl: utility
```
Create a compute pass consisting of the four compute shaders (in the order shown above) with pipeline barriers between each of them:
```
VkMemoryBarrier memoryBarrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
```
Set the global invocation sizes of the four shaders:
```cpp
lbvh_morton_codes: (NUM_ELEMENTS, 1, 1) // (x,y,z global invocation sizes)
lbvh_single_radixsort: (256, 1, 1) // 256=WORKGROUP_SIZE defined in lbvh_single_radix_sort.comp, i.e. we just want to launch a single work group
lbvh_hierarchy: (NUM_ELEMENTS, 1, 1)
lbvh_bounding_boxes: (NUM_ELEMENTS, 1, 1)
```

<a name="buffers"></a>
### Buffers
Create the following five buffers and assign them to the following sets and indices of your compute pass:

| buffer | size (bytes) | initialize      | (set,index)       |
| - | - |-----------------|-------------------|
| m_elementsBuffer | NUM_ELEMENTS * sizeof(Element) | vector of elements | (0,1),(2,1)       |
| m_mortonCodeBuffer | NUM_ELEMENTS * sizeof(MortonCodeElement) | - | (0,0),(1,0),(2,0) |
| m_mortonCodePingPongBuffer | NUM_ELEMENTS * sizeof(MortonCodeElement) | - | (1,1)             |
| m_LBVHBuffer | NUM_LBVH_ELEMENTS * sizeof(LBVHNode) | - | (2,2),(3,0)       |
| m_LBVHConstructionInfoBuffer | NUM_LBVH_ELEMENTS * sizeof(LBVHConstructionInfo) | - | (2,3),(3,1)       |

Use `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` and `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.

<a name="push--constants"></a>
### Push Constants
Define the following push constant structs for the four shaders and set their data:
```cpp
struct PushConstantsMortonCodes {
    uint32_t g_num_elements; // = NUM_ELEMENTS
    float g_min_x; // (*)
    float g_min_y;
    float g_min_z;
    float g_max_x;
    float g_max_y;
    float g_max_z;
};

struct PushConstantsRadixSort {
    uint32_t g_num_elements; // = NUM_ELEMENTS
};

struct PushConstantsHierarchy {
    uint32_t g_num_elements; // = NUM_ELEMENTS
    uint32_t g_absolute_pointers; // 1 or 0 (**)
};

struct PushConstantsBoundingBoxes {
    uint32_t g_num_elements; // = NUM_ELEMENTS
    uint32_t g_absolute_pointers; // 1 or 0 (**)
};
```
(*) AABB that contains the entire model. Based on their floating point positions, each primitive is assigned an integer morton code, i.e. the position is discretized. The provided AABB defines the range of possible floating point positions for the mapping. The tighter the range, the more distinct morton codes are in a certain interval. To ensure the largest number of distinct morton codes (for presumably better BVH quality), choose the AABB as tight as possible, i.e. the union of all AABBs of the elements/primitives. However, since the builder can handle duplicate morton codes of two elements (when their different floating point position is mapped to the same morton code), you may define a larger AABB, which can reduce building time.

(**) The builder supports absolute (1) and relative (0) child pointers. The resulting LBVH is stored as an array of (LBVH)nodes. Absolute child pointers point directly to the index in the array. Relative pointers store the relative shift from the index of the current parent node to the index of the child node, e.g. the absolute index of the current node is `i` and the stored relative pointer to the child is `j`, then the absolute index into the array of the child is `i + j`. Note that the relative pointer may be negative to indicate that the child node is in front of the current node in the array.

<a name="execute-"></a>
### Execute 
Execute the compute pass. Wait for the compute queue to idle. The result is in the `m_LBVHBuffer` buffer.

<a name="screenshot"></a>
## Screenshot
The example implementation writes the constructed LBVH of the [Stanford Dragon](http://graphics.stanford.edu/data/3Dscanrep/) model to a csv file which, for example, can be visualized with my [BVHVisualization](https://github.com/MircoWerner/BVHVisualization).

![img bvh visualization](https://github.com/MircoWerner/BVHVisualization/blob/main/resources/bvhexample/lbvh_visualization.png?raw=true)
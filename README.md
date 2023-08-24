# Vulkan LBVH (Linear Bounding Volume Hierarchy)

**GPU LBVH builder** implemented in **Vulkan** and **GLSL**.

The implementation is based on the following paper:
- [Tero Karras. 2012. Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. In Eurographics/ ACM SIGGRAPH Symposium on High Performance Graphics, The Eurographics Association. DOI:https://doi.org/10.2312/EGGH/HPG12/033-037](https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf)

and inspired by / based on the following repositories and blog posts:
- [LBVH implementation: CUDA by ToruNiina](https://github.com/ToruNiina/lbvh)
- [LBVH blog post: Tree Construction on the GPU by Tero Karras](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)
- [RadixSort implementation: Embree by Intel](https://github.com/embree/embree/blob/v4.0.0-ploc/kernels/rthwif/builder/gpu/sort.h) (radix sort part of lbvh building algorithm)

## Table of Contents
- Example Usage (reference implementation in Vulkan)
  - [Compile / Run](#compile--run)
- Own Usage (how to use the LBVH builder in your own Vulkan project)

<a name="compile--run"></a>
## Example Usage
This repository contains a reference implementation to show how to use the provided shaders for the LBVH builder.

### Compile / Run
Requirements: Vulkan, glm

```bash 
git clone --recursive git@github.com:MircoWerner/LBVH.git
cd LBVH
mkdir build
cmake ..
make
./lbvhexample
```

## Own Usage
Explanation how to use the LBVH builder in your own Vulkan project.

### Struct Definition
Define the following structs:
```cpp
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
```

### Model Loading
Load some model containing `NUM_ELEMENTS` primitives. The LBVH will contain `NUM_LBVH_ELEMENTS = NUM_ELEMENTS + NUM_ELEMENTS - 1;` nodes after building.
Define a vector/array containing `Element` structs for all primitives. This vector/array will be uploaded to the input (element) buffer for building.

### Shaders / Compute Pass
Copy the following shaders to your project:
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
lbvh_morton_codes: (NUM_ELEMENTS, 1, 1); // (x,y,z global invocation sizes)
lbvh_single_radixsort: (1, 1, 1);
lbvh_hierarchy: (NUM_ELEMENTS, 1, 1);
lbvh_bounding_boxes: (NUM_ELEMENTS, 1, 1);
```

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

### Execute 
Execute the compute pass. Wait for the compute queue to idle. The result is in the `m_LBVHBuffer` buffer.
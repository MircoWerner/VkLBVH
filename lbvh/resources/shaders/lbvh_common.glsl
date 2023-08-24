#ifndef LBVH_COMMONG_GLSL
#define LBVH_COMMONG_GLSL

#define INVALID_POINTER 0x0

// input for the builder (normally a triangle or some other kind of primitive); it is necessary to allocate and fill the buffer
struct Element {
    uint primitiveIdx;// the id of the primitive; this primitive id is copied to the leaf nodes of the  LBVHNode
    float aabbMinX;// aabb of the primitive
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

// output of the builder; it is necessary to allocate the (empty) buffer
struct LBVHNode {
    int left;// pointer to the left child or INVALID_POINTER in case of leaf
    int right;// pointer to the right child or INVALID_POINTER in case of leaf
    uint primitiveIdx;// custom value that is copied from the input Element or 0 in case of inner node
    float aabbMinX;// aabb of the node
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

// only used on the GPU side during construction; it is necessary to allocate the (empty) buffer
struct MortonCodeElement {
    uint mortonCode;// key for sorting
    uint elementIdx;// pointer into element buffer
};

// only used on the GPU side during construction; it is necessary to allocate the (empty) buffer
struct LBVHConstructionInfo {
    uint parent;// pointer to the parent
    int visitationCount;// number of threads that arrived
};

#endif
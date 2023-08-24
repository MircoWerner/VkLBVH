#ifndef LBVH_COMMONG_GLSL
#define LBVH_COMMONG_GLSL

struct Element {
    uint primitiveIdx;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

struct MortonCodeElement {
    uint mortonCode; // key for sorting
    uint elementIdx; // pointer into element buffer
};

struct LBVHNode {
    int left;
    int right;
    uint primitiveIdx;
    float aabbMinX;
    float aabbMinY;
    float aabbMinZ;
    float aabbMaxX;
    float aabbMaxY;
    float aabbMaxZ;
};

struct LBVHConstructionInfo {
    uint parent;
    int visitationCount;
};

#endif
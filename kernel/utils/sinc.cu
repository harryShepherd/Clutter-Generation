#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ inline float sincf(float x)
{
    if (x == 0.0f) return 1.0f;

    return sinf(x) / x;
}
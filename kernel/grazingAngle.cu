#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "maths_constants.h"

__global__ GrazingAngle(
    float altitude,
    float radius_of_earth,
    size_t total_range_rings,
    float* isorange_rings_input,
    float* isorange_rings_output
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid > total_range_rings) continue; // do not exceed vector limits

    float slant_range = (isorange_rings_input[tid] + isorange_rings_input[tid + 1]) / 2.0f;

    float grazing_angle = 
        (pow(slant_range, 2) - pow(altitude + radius_of_earth, 2) + pow(radius_of_earth, 2)) /
        (2.0f * slant_range * radius_of_earth);
    
    if (abs(grazing_angle) < 1)
    {
        grazing_angle = -(asin(grazing_angle));
    }
    else
    {
        grazing_angle = CUDART_PI_F / 2.0f;
    }

    isorange_rings_output[tid] = grazing_angle;
}
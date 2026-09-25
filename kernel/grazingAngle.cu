#include "grazingAngle.cuh"

__global__ void GrazingAngle(
    float altitude,
    float radius_of_earth,
    size_t total_range_rings,
    float* isorange_rings_input,
    float* grazing_angles
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid > total_range_rings) return; // do not exceed vector limits

    float slant_range = (isorange_rings_input[tid] + isorange_rings_input[tid + 1]) / 2.0f;

    if (slant_range < altitude)
    {
        // slant range can never be less than altitude
        grazing_angles[tid] = 0.0f;
        return;
    }

    float grazing_angle = 
        (powf(slant_range, 2) - powf(altitude + radius_of_earth, 2) + powf(radius_of_earth, 2)) /
        (2.0f * slant_range * radius_of_earth);
    
    if (abs(grazing_angle) < 1)
    {
        grazing_angle = -(asin(grazing_angle));
    }
    else
    {
        grazing_angle = CUDART_PIO2_F;
    }

    grazing_angles[tid] = grazing_angle;
}
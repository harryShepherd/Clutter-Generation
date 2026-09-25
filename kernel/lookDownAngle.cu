#include "lookDownAngle.cuh"

__global__ void LookDownAngle(
    float altitude,
    float radius_of_earth,
    size_t total_range_rings,
    float* isorange_rings_inputs,
    float* look_down_angles
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid > total_range_rings) return; // do not exceed vector limits

    float slant_range = (isorange_rings_inputs[tid] + isorange_rings_inputs[tid + 1]) / 2.0f;

    if (slant_range < altitude)
    {
        // slant range can never be less than altitude
        look_down_angles[tid] = 0.0f;
        return;
    }

    float look_down_angle = (
        (powf(slant_range, 2) + powf(altitude + radius_of_earth, 2) - powf(radius_of_earth, 2)) /
        (2.0f * slant_range * (altitude + radius_of_earth)));

    if (abs(look_down_angle) < 1)
    {
        look_down_angle = asin(look_down_angle);
    }
    else
    {
        look_down_angle = CUDART_PIO2_F;
    }

    look_down_angles[tid] = look_down_angle;
}
#include "patchArea.cuh"

__global__ void calculatePatchArea(
    float altitude,
	float radius_of_earth,
	float delta_azimuth,
	size_t total_range_rings,
	float* isorange_rings_input,
	float* isorange_rings_output
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid > total_range_rings - 1) return; // do not exceed vector limits

    float r1 = isorange_rings_input[tid];
    float r2 = isorange_rings_input[tid + 1];

    // calculate area of clutter patch
    float area =
        0.5f *
        delta_azimuth *
        (r2 * r2 - r1 * r1) *
        radius_of_earth /
        (radius_of_earth + altitude);

    isorange_rings_output[tid] = area;
}
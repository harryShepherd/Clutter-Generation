#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include <math.h>

__global__ void GrazingAngle(
    float altitude,
    float radius_of_earth,
    size_t total_range_rings,
    float* isorange_rings,
    float* grazing_angle
);

__device__ void sm_grazingAngle(
    const float altitude,
    const float radius_of_earth,
    const size_t total_range_rings,
    float* isorange_rings_input,
    float* grazing_angle_output
);
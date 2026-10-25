#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

__global__ void LookDownAngle(
    float altitude,
    float radius_of_earth,
    size_t total_range_rings,
    float* isorange_rings_inputs,
    float* look_down_angles
);

__device__ void sm_lookDownAngle(
    const float altitude,
    const float radius_of_earth,
    const size_t total_range_rings,
    float* isorange_rings_input,
    float* look_down_angles_output
);
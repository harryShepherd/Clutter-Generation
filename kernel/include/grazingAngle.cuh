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
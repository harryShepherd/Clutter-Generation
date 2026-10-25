#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ __inline__ float patchArea(
    const float ring1,
    const float ring2,
    const float altitude,
    const float radius_of_earth,
    const float delta_azimuth
)
{
    // calculate area of clutter patch
    return 0.5f * delta_azimuth *
        (ring2 * ring2 - ring1 * ring1) *
        radius_of_earth / (radius_of_earth + altitude);
}
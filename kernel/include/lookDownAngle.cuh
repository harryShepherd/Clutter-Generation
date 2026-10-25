#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_constants.h>

#include <math.h>

__device__ __inline__ float lookDownAngle(
    const float altitude,
    const float radius_of_earth,
    const float slant_range
)
{
    if (slant_range < altitude)
    {
        // slant range can never be less than altitude
        return 0.0f;
    }

    float look_down_angle = (
        (powf(slant_range, 2) + powf(altitude + radius_of_earth, 2) - powf(radius_of_earth, 2)) /
        (2.0f * slant_range * (altitude + radius_of_earth)));

    if (fabsf(look_down_angle) < 1)
    {
        return asin(look_down_angle);
    }

    return CUDART_PIO2_F;
}
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

__device__ __inline__ float Frequency(
    const float az,
    const float el,
    const float vel,
    const float wavelength
)
{
    return (-2.0f * vel * cosf(az) * cosf(el)) / wavelength;
}
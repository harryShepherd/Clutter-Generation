#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

__device__ float Frequency(
    const float az,
    const float el,
    const float vel,
    const float wavelength
);
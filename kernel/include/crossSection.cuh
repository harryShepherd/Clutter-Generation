#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ inline float ClutterCrossSection(
    float gamma,
    float grazing_angle,
    float patch_area
);
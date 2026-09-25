#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float AntennaGain(
    float beam_width,
    float azimuth,
    float elevation);
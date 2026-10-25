#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__device__ __inline__ float sincf(float x)
{
    if (x == 0.0f) return 1.0f;

    return sinf(x) / x;
}

__device__ __inline__ float antennaGain(
    float beam_width,
    float azimuth,
    float elevation)
{
    float az_sinc = sincf(azimuth / beam_width);
    float el_sinc = sincf(elevation / beam_width);

    float gain = az_sinc * az_sinc * el_sinc * el_sinc;

    return gain;
}
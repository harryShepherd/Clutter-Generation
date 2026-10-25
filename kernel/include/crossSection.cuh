#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

__device__ __inline__ float MeanClutterCrossSection(
    const float scattering_coefficient,
    const float grazing_angle
)
{
    return scattering_coefficient * sinf(grazing_angle);
}

__device__ __inline__ float GenerateRandomSigma0(
    const float mean_sigma0,
    const float random_0_1)
{
    return -mean_sigma0 * logf(random_0_1);
}

__device__ __inline__ float crossSection(
    const float gamma,
    const float grazing_angle,
    const float patch_area,
    const float random_0_1
)
{
    float mean_sigma0 = MeanClutterCrossSection(gamma, grazing_angle);
    float sigma0 = GenerateRandomSigma0(mean_sigma0, random_0_1);

    return sigma0 * patch_area;
}
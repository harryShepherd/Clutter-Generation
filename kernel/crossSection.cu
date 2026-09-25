#include "crossSection.cuh"

__device__ inline float MeanClutterCrossSection(
	float scattering_coefficient,
	float grazing_angle
)
{
	return scattering_coefficient * sinf(grazing_angle);
}

__device__ inline float GenerateRandomSigma0(float mean_sigma0)
{
    float rand = 0.45f; // TODO: Implement CUDA RNG

    return -mean_sigma0 * logf(rand);
}

__device__ inline float ClutterCrossSection(
    float gamma,
    float grazing_angle,
    float patch_area
)
{
    float mean_sigma0 = MeanClutterCrossSection(gamma, grazing_angle);
    float sigma0 = GenerateRandomSigma0(mean_sigma0);
    float cross_section = sigma0 * patch_area;
    
    return cross_section;
}
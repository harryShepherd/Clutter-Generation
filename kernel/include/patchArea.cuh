#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void calculatePatchArea(
	float altitude,
	float radius_of_earth,
	float delta_azimuth,
	size_t total_range_rings,
	float* isorange_rings_input,
	float* isorange_rings_output
);

__device__ void sm_calculatePatchArea(
	const float altitude,
	const float radius_of_earth,
	const float delta_azimuth,
	const size_t total_range_rings,
	float* isorange_rings_input,
	float* isorange_rings_output
);

__device__ void sm_createRangeRings(
	const size_t total_range_rings,
	const float range_sample_spacing,
	float* isorange_rings_output
);
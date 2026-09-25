#pragma once 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include <cmath>

__device__ float patchPower(
	float transmitted_power,
	float antenna_gain,
	float cross_section,
	float wavelength,
	float slant_range,
	float signal_loss
);
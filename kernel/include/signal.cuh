#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "math_constants.h"
#include <cmath>

__global__ void CalculateSignal(
    const size_t total_range_rings,
    const size_t total_azimuth_bins,
    const size_t total_pulses,
    const float azimuth_min,
    const float delta_azimuth,
    const int prf,
    const float vel,
    const float gamma,
    const float beam_width,
    const float trans_power,
    const float wavelength,
    const float signal_loss,
    float* isorange_rings,
    float* patch_areas,
    float* look_down_angles,
    float* grazing_angles,
    cuFloatComplex* signal_output
);
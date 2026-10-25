#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "math_constants.h"
#include <cmath>

// Parameters of each patch that remain unchanged between pulses
struct patchParams
{
    float amplitude;            // sqrt of the power
    float doppler_frequency;    //
    float random_phase;         // phi_0
};

// Calculate patch-specific values. Writes out to a patchParams array
__global__ void patchSetup(
    const int total_range_rings,
    const int total_azimuth_bins,
    const float prf,
    const float delta_azimuth,
    const float radius_of_earth,
    const float altitude,
    const float velocity,
    const float beam_width,
    const float transmitted_power,
    const float wavelength,
    const float signal_loss,
    const float azimuth_min,
    const float gamma,
    const unsigned long long rand_seed,
    const float* isorange_rings_in,
    patchParams* __restrict__ out);

__global__ void pulseSamples(
    const patchParams* __restrict__ patch_params,
    const int total_pulses,
    cuFloatComplex* out);
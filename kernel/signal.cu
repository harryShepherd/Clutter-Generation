#include "signal.cuh"
#include "frequency.cuh"
#include "crossSection.cuh"
#include "antennaGain.cuh"
#include "patchPower.cuh"
#include "patchArea.cuh"
#include "grazingAngle.cuh"
#include "lookDownAngle.cuh"

#include <curand_kernel.h>

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
    patchParams* __restrict__ out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total_range_rings * total_azimuth_bins) return;

    // find out with range ring and azimuth bin we're working on
    int range_ring = tid / total_azimuth_bins;
    int azimuth_bin = tid % total_azimuth_bins;

    float ring1 = isorange_rings_in[tid];
    float ring2 = isorange_rings_in[tid + 1];
    float slant_range = (ring1 + ring2) / 2.0f;

    // get some information about the patch we're working on
    float patch_area = patchArea(ring1, ring2, altitude, radius_of_earth, delta_azimuth);
    float patch_grazing_angle = grazingAngle(altitude, radius_of_earth, slant_range);
    float patch_look_down_angle = lookDownAngle(altitude, radius_of_earth, slant_range);
    float patch_azimuth = (azimuth_min + (delta_azimuth * azimuth_bin)) + (delta_azimuth / 2.0f);

    // random number generation
    curandStatePhilox4_32_10_t state;
    curand_init(rand_seed, tid, 0, &state);
    double2 rand = curand_uniform2_double(&state); // rand.x -> sigma_0, rand.y -> phi_0

    float patch_cross_section = crossSection(gamma, patch_grazing_angle, patch_area, rand.x);
    float antenna_gain = antennaGain(beam_width, patch_azimuth, patch_look_down_angle);

    float patch_power = patchPower(transmitted_power, antenna_gain,  patch_cross_section, wavelength, slant_range, signal_loss);

    float doppler_frequency = Frequency(patch_azimuth, patch_look_down_angle, velocity, wavelength);

    // create patch parameters
    patchParams params{
        sqrtf(patch_power),         // amplitude
        doppler_frequency / prf,    // doppler frequency per cycle
        rand.y                      // phi_0
    };

    out[tid] = params;
}

__global__ void pulseSamples(
    const patchParams* __restrict__ patch_params,
    const int total_pulses,
    cuFloatComplex* out)
{
    // read-only
    const patchParams patch = patch_params[blockIdx.x];

    cuFloatComplex* c = out + static_cast<size_t>(blockIdx.x) * total_pulses;

    for (int pulse = threadIdx.x; pulse < total_pulses; pulse += blockDim.x)
    {
        // fmaf computes (x * y) + z in one instruction
        // this was previously written as:
        //      2.0f * CUDART_PI_F * doppler_frequency * transmitted_pulse_time + phi_0
        // we are doing the same, except having removed 2pi
        float cycle = fmaf(patch.doppler_frequency, pulse, patch.random_phase);
        cycle -= floorf(cycle);

        float s, c;
        // 2.0f * cycle is where we re-add 2pi from the previous equation
        sincospif(2.0f * cycle, &s, &c);

        out[pulse] = make_cuFloatComplex(patch.amplitude * c, -patch.amplitude * s);
    }
}
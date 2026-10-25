#include "signal.cuh"
#include "frequency.cuh"
#include "crossSection.cuh"
#include "antennaGain.cuh"
#include "patchPower.cuh"
#include "patchArea.cuh"
#include "grazingAngle.cuh"
#include "LookDownAngle.cuh"

#include <cstdio>

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
    float* random_numbers,
    cuFloatComplex* signal_output
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= (total_range_rings - 1) * total_azimuth_bins * total_pulses) return;

    int current_pulse = tid % total_pulses;
    int current_az_bin = (tid / total_pulses) % total_azimuth_bins;
    int current_range_ring = (tid / total_pulses) / total_azimuth_bins;

    // get some information about the patch we're working on
    float patch_area = patch_areas[current_range_ring];
    float patch_look_down_angle = look_down_angles[current_range_ring];
    float patch_grazing_angle = grazing_angles[current_range_ring];
    float patch_azimuth = (azimuth_min + (delta_azimuth * current_az_bin)) + (delta_azimuth / 2.0f);

    // calculate slant range
    float slant_range = (isorange_rings[current_range_ring] + isorange_rings[current_range_ring + 1]) / 2.0f;

    // calculate cross section
    float cross_section = ClutterCrossSection(gamma, patch_grazing_angle, patch_area);

    // calculate antenna gain
    float antenna_gain = AntennaGain(beam_width, patch_azimuth, patch_look_down_angle);
    
    // calculate the power returned by the clutter patch
    float patch_power = patchPower(
        trans_power,
        antenna_gain,
        cross_section,
        wavelength,
        slant_range,
        signal_loss
    );

    float phi_0 = random_numbers[tid]; 

    float amplitude = sqrtf(patch_power);

    float doppler_frequency = Frequency(
        patch_azimuth,
        patch_look_down_angle,
        vel,
        wavelength
    );

    float transmitted_pulse_time = current_pulse * (1.0f / (float)prf);

    float x = 2.0f * CUDART_PI_F * doppler_frequency * transmitted_pulse_time + phi_0;

    float sinx = sinf(x);
    float cosx = cosf(x);

    cuFloatComplex c = make_cuFloatComplex(amplitude * cosx, -amplitude * sinx);

    signal_output[tid] = c;
}

__global__ void CalculateSignalSharedMem(
    const size_t total_range_rings,
    const size_t total_azimuth_bins,
    const size_t total_pulses,
    const float altitude,
    const float radius_of_earth,
    const float azimuth_min,
    const float delta_azimuth,
    const float range_sample_spacing,
    const int prf,
    const float vel,
    const float gamma,
    const float beam_width,
    const float trans_power,
    const float wavelength,
    const float signal_loss,
    float* random_numbers,
    cuFloatComplex* signal_output
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (total_range_rings - 1) * total_azimuth_bins * total_pulses) return;

    int current_pulse = tid % total_pulses;
    int current_az_bin = (tid / total_pulses) % total_azimuth_bins;
    int current_range_ring = (tid / total_pulses) / total_azimuth_bins;
    
    // create isorange rings
    extern __shared__ float sm_range_rings[];
    sm_createRangeRings(
        total_range_rings,
        range_sample_spacing,
        sm_range_rings
    );
    
    // calculate patch areas
    extern __shared__ float sm_patch_areas[];
    sm_calculatePatchArea(
        altitude,
        radius_of_earth,
        delta_azimuth,
        total_range_rings,
        sm_range_rings,
        sm_patch_areas
    );

    // calculate grazing angles
    extern __shared__ float sm_grazing_angles[];
    sm_grazingAngle(
        altitude,
        radius_of_earth,
        total_range_rings,
        sm_range_rings,
        sm_grazing_angles
    );

    // calculate look down angles
    extern __shared__ float sm_look_down_angles[];
    sm_lookDownAngle(
        altitude,
        radius_of_earth,
        total_range_rings,
        sm_range_rings,
        sm_look_down_angles
    );

    float patch_area = sm_patch_areas[current_range_ring];
    float patch_look_down_angle = sm_look_down_angles[current_range_ring];
    float patch_grazing_angle = sm_grazing_angles[current_range_ring];
    float patch_azimuth = (azimuth_min + (delta_azimuth * current_az_bin)) + (delta_azimuth / 2.0f);

    // calculate slant range
    float slant_range = (sm_range_rings[current_range_ring] + sm_range_rings[current_range_ring + 1]) / 2.0f;

    // calculate cross section
    float cross_section = ClutterCrossSection(gamma, patch_grazing_angle, patch_area);

    // calculate antenna gain
    float antenna_gain = AntennaGain(beam_width, patch_azimuth, patch_look_down_angle);

    // calculate the power returned by the clutter patch
    float patch_power = patchPower(
        trans_power,
        antenna_gain,
        cross_section,
        wavelength,
        slant_range,
        signal_loss
    );

    float phi_0 = random_numbers[tid];

    float amplitude = sqrtf(patch_power);

    float doppler_frequency = Frequency(
        patch_azimuth,
        patch_look_down_angle,
        vel,
        wavelength
    );

    float transmitted_pulse_time = current_pulse * (1.0f / (float)prf);

    float x = 2.0f * CUDART_PI_F * doppler_frequency * transmitted_pulse_time + phi_0;

    float sinx = sinf(x);
    float cosx = cosf(x);

    cuFloatComplex c = make_cuFloatComplex(amplitude * cosx, -amplitude * sinx);

    signal_output[tid] = c;
}
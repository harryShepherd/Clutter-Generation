#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <fstream>

#include "antennaGain.cuh"
#include "crossSection.cuh"
#include "frequency.cuh"
#include "grazingAngle.cuh"
#include "lookDownAngle.cuh"
#include "patchArea.cuh"
#include "signal.cuh"

#include "cudaErr.cuh"

int main()
{    
    // parameters

    const float transmitted_power = 15000.0f;  // W
    const float wavelength = 0.03f;            // m
    const float PRF = 15000.0f;                // Hz

    const float beam_width = 5.0f;             // degs

    const int coherent_pulses = 64;
    const int range_samples = 100;

    const float range_sample_spacing = 100.0f; // m

    const float altitude = 5250.0f;             // m
    const float ownship_velocity = 500.0f;      // m/s

    const float radius_of_earth = 6.371e6f;     // m

    const float look_down_boresight = 45.0f * CUDART_PI / 180.0f;

    const float look_aside_boresight = 1.0f * CUDART_PI / 180.0f;

    const float delta_azimuth = 0.1f * CUDART_PI / 180.0f;

    float azimuth_min = -5.0f * CUDART_PI / 180.0f; // -5 degs
    float azimuth_max = 5.0f * CUDART_PI / 180.0f; // +5 degs

    const float gamma = 0.1f;

    const float signal_loss = 1.0f; // 1 means no signal loss
    
    int total_isorange_rings = 200;

    std::vector<float> isorange_rings(total_isorange_rings), patches(total_isorange_rings), grazing_angles(total_isorange_rings), look_down_angles(total_isorange_rings);
    float* dev_isorange_rings, * dev_patches, * dev_grazing_angles, * dev_look_down_angles;

    for (int i = 0; i < total_isorange_rings; ++i)
    {
        isorange_rings[i] = i * range_sample_spacing;
    }

    size_t total_az_patches = static_cast<size_t>((azimuth_max - azimuth_min) / delta_azimuth);
    size_t total_signals = (total_isorange_rings * total_az_patches * coherent_pulses);

    std::vector<cuFloatComplex> signals(total_signals);
    cuFloatComplex* dev_signals;

    // end of parameters

    // allocate memory on device
    cudaErrChk(cudaMalloc(&dev_isorange_rings, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_patches, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_grazing_angles, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_look_down_angles, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_signals, total_signals * sizeof(cuFloatComplex)));

    // copy isorange rings to device
    cudaErrChk(cudaMemcpy(dev_isorange_rings, isorange_rings.data(), total_isorange_rings * sizeof(float), cudaMemcpyHostToDevice));

    // calculate clutter patch areas
    size_t threads = 124;
    size_t blocks = std::ceilf(static_cast<float>(total_isorange_rings) / static_cast<float>(threads));

    calculatePatchArea<<<blocks, threads>>>(altitude, radius_of_earth, delta_azimuth, total_isorange_rings, dev_isorange_rings, dev_patches);

    // calculate grazing angles
    GrazingAngle<<<blocks, threads>>>(altitude, radius_of_earth, total_isorange_rings, dev_isorange_rings, dev_grazing_angles);

    // calculate look down angles
    LookDownAngle<<<blocks, threads>>>(altitude, radius_of_earth, total_isorange_rings, dev_isorange_rings, dev_look_down_angles);

    // calculate how many signals we need to generate
    blocks = std::ceilf(static_cast<float>(total_signals) / static_cast<float>(threads));

    CalculateSignal<<<blocks, threads>>>(
        total_isorange_rings,
        total_az_patches,
        coherent_pulses,
        azimuth_min,
        delta_azimuth,
        PRF,
        ownship_velocity,
        gamma,
        beam_width,
        transmitted_power,
        wavelength,
        signal_loss,
        dev_isorange_rings,
        dev_patches,
        dev_look_down_angles,
        dev_grazing_angles,
        dev_signals);

    cudaErrChk(cudaPeekAtLastError());
    //cudaErrChk(cudaDeviceSynchronize());

    cudaErrChk(cudaMemcpy(signals.data(), dev_signals, total_signals * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    cudaErrChk(cudaFree(dev_isorange_rings));
    cudaErrChk(cudaFree(dev_patches));
    cudaErrChk(cudaFree(dev_grazing_angles));
    cudaErrChk(cudaFree(dev_look_down_angles));
    cudaErrChk(cudaFree(dev_signals));

    std::cout << "Complete" << std::endl;

    std::cout << "Writing to file" << std::endl;

    std::ofstream output_file("clutter_signals.txt");

    if (!output_file)
    {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    for (cuFloatComplex c : signals)
    {
        output_file << c.x << " " << c.y << std::endl;
    }

    std::cout << "Writing Complete" << std::endl;

    output_file.close();
}
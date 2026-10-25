#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>

#include "signal.cuh"
#include "cudaErr.cuh"

int main()
{    
    const float transmitted_power = 15000.0f;   // W
    const float wavelength = 0.03f;             // m
    const float PRF = 15000.0f;                 // Hz

    const float beam_width = 5.0f;              // degs

    const int coherent_pulses = 64;
    const int range_samples = 100;

    const float range_sample_spacing = 100.0f;  // m

    const float altitude = 5250.0f;             // m
    const float ownship_velocity = 500.0f;      // m/s

    const float radius_of_earth = 6.371e6f;     // m

    constexpr float look_down_boresight = 45.0f * CUDART_PI / 180.0f;

    constexpr float look_aside_boresight = 1.0f * CUDART_PI / 180.0f;

    constexpr float delta_azimuth = 0.1f * CUDART_PI / 180.0f;

    constexpr float azimuth_min = -5.0f * CUDART_PI / 180.0f; // -5 degs
    constexpr float azimuth_max = 5.0f * CUDART_PI / 180.0f; // +5 degs

    const float gamma = 0.1f;

    const float signal_loss = 1.0f; // 1 means no signal loss

    const unsigned long long seed = 1;
    
    const int total_isorange_rings = 200;

    std::vector<float> isorange_rings(total_isorange_rings),
        patches(total_isorange_rings),
        grazing_angles(total_isorange_rings),
        look_down_angles(total_isorange_rings);

    // populate ring edges
    // this is just sample data
    for (int i = 0; i < total_isorange_rings; ++i)
    {
        isorange_rings[i] = i * range_sample_spacing;
    }

    const int total_az_patches = static_cast<int>((azimuth_max - azimuth_min) / delta_azimuth);
    const int total_patches = total_isorange_rings * total_az_patches;
    const int total_signals = (total_isorange_rings * total_az_patches * coherent_pulses);
    const int total_signal_bytes = sizeof(cuFloatComplex) * total_signals;

    cudaErrChk(cudaFree(0));
    
    float* dev_isorange_rings;
    patchParams* dev_patch_params;
    cuFloatComplex* dev_signals;
    cuFloatComplex* signals; // pinned memory

    // allocate memory on device
    cudaErrChk(cudaMalloc(&dev_isorange_rings, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_patch_params, total_patches * sizeof(patchParams)));
    cudaErrChk(cudaMalloc(&dev_signals, total_signal_bytes));
    cudaErrChk(cudaMallocHost(&signals, total_signal_bytes));

    // copy isorange rings to device
    cudaErrChk(cudaMemcpy(dev_isorange_rings, isorange_rings.data(), total_isorange_rings * sizeof(float), cudaMemcpyHostToDevice));

    // calculate clutter patch threads/blocks
    const int patch_threads = 128;
    const int patch_blocks = (total_patches + patch_threads - 1) / patch_threads;

    const int pulse_threads = std::min(256, ((coherent_pulses + 31) / 32) * 32);

    // function to run kernels
    auto generate = [&]()
        {
            // patch setup kernel
            patchSetup << <patch_blocks, patch_threads >> > (
                total_isorange_rings,
                total_az_patches,
                PRF,
                delta_azimuth,
                radius_of_earth,
                altitude,
                ownship_velocity,
                beam_width,
                transmitted_power,
                wavelength,
                signal_loss,
                azimuth_min,
                gamma,
                seed,
                dev_isorange_rings,
                dev_patch_params);

            // sample pulses
            pulseSamples << <total_patches, pulse_threads >> > (
                dev_patch_params,
                coherent_pulses,
                dev_signals);
        };

    // warm-up kernels
    generate();
    cudaErrChk(cudaPeekAtLastError());
    cudaErrChk(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaErrChk(cudaEventCreate(&start));
    cudaErrChk(cudaEventCreate(&stop));

    cudaErrChk(cudaEventRecord(start));
    generate();
    cudaErrChk(cudaEventRecord(stop));

    cudaErrChk(cudaPeekAtLastError());
    cudaErrChk(cudaEventSynchronize(stop));

    cudaErrChk(cudaFree(dev_isorange_rings));

    std::cout << "Calculation complete" << std::endl;

    float kernel_ms = 0.0f;
    cudaErrChk(cudaEventElapsedTime(&kernel_ms, start, stop));

    std::cout << "Kernels took " << kernel_ms << "ms" << std::endl;

    std::cout << "Copying to host" << std::endl;

    auto copy_start = std::chrono::steady_clock::now();

    cudaErrChk(cudaMemcpy(signals, dev_signals, total_signal_bytes, cudaMemcpyDeviceToHost));
    cudaErrChk(cudaFree(dev_signals));

    auto copy_stop = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> copy_ms = copy_stop - copy_start;


    std::cout << "Writing to file" << std::endl;

    std::ofstream output_file("clutter_signals.txt");

    if (!output_file)
    {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    for (int i = 0; i < total_signals; ++i)
    {
        output_file << signals[i].x << " " << signals[i].y << std::endl;
    }

    std::cout << "Writing Complete" << std::endl;

    output_file.close();

    cudaErrChk(cudaFreeHost(signals));
}
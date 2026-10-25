#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <fstream>
#include <chrono>

#include "curand.h"

#include "antennaGain.cuh"
#include "crossSection.cuh"
#include "frequency.cuh"
#include "grazingAngle.cuh"
#include "lookDownAngle.cuh"
#include "patchArea.cuh"
#include "signal.cuh"

#include "cudaErr.cuh"

int clutter()
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

    std::vector<float> isorange_rings(total_isorange_rings),
        patches(total_isorange_rings),
        grazing_angles(total_isorange_rings),
        look_down_angles(total_isorange_rings);

    float* dev_isorange_rings,
        * dev_patches,
        * dev_grazing_angles,
        * dev_look_down_angles,
        * dev_random_numbers;

    for (int i = 0; i < total_isorange_rings; ++i)
    {
        isorange_rings[i] = i * range_sample_spacing;
    }

    size_t total_az_patches = static_cast<size_t>((azimuth_max - azimuth_min) / delta_azimuth);
    size_t total_signals = (total_isorange_rings * total_az_patches * coherent_pulses);

    std::vector<cuFloatComplex> signals(total_signals);
    cuFloatComplex* dev_signals;

    // start timer
    auto start = std::chrono::steady_clock::now();

    // end of parameters

    // allocate memory on device
    cudaErrChk(cudaMalloc(&dev_isorange_rings, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_patches, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_grazing_angles, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_look_down_angles, total_isorange_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&dev_signals, total_signals * sizeof(cuFloatComplex)));
    cudaErrChk(cudaMalloc(&dev_random_numbers, total_signals * sizeof(float)));

    int total_mem = total_isorange_rings * sizeof(float);       // isorange rings
    std::cout << "[GPU] Total Memory Usage Isorange Rings: " << total_mem << " bytes" << std::endl;

    total_mem = total_isorange_rings * sizeof(float);          // patch areas
    std::cout << "[GPU] Total Memory Usage Patch Areas: " << total_mem << " bytes" << std::endl;

    total_mem = total_isorange_rings * sizeof(float);          // grazing angles
    std::cout << "[GPU] Total Memory Usage Grazing Angles: " << total_mem << " bytes" << std::endl;

    total_mem = total_isorange_rings * sizeof(float);          // look down angles
    std::cout << "[GPU] Total Memory Usage Look Down Angles: " << total_mem << " bytes" << std::endl;

    //total_mem += total_signals * sizeof(cuFloatComplex);      // signals

    total_mem = total_signals * sizeof(float);                 // random numbers
    std::cout << "[GPU] Total Memory Usage Random Numbers: " << total_mem << " bytes" << std::endl;


    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "[GPU] Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;

    // copy isorange rings to device
    cudaErrChk(cudaMemcpy(dev_isorange_rings, isorange_rings.data(), total_isorange_rings * sizeof(float), cudaMemcpyHostToDevice));

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    std::cout << "[CPU] Memory Allocation Took " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();

    // calculate clutter patch areas
    size_t threads = 124;
    size_t blocks = std::ceilf(static_cast<float>(total_isorange_rings) / static_cast<float>(threads));

    // cuda execution time measuring
    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);

    cudaEventRecord(c_start);

    calculatePatchArea << <blocks, threads >> > (altitude, radius_of_earth, delta_azimuth, total_isorange_rings, dev_isorange_rings, dev_patches);

    // calculate grazing angles
    GrazingAngle << <blocks, threads >> > (altitude, radius_of_earth, total_isorange_rings, dev_isorange_rings, dev_grazing_angles);

    // calculate look down angles
    LookDownAngle << <blocks, threads >> > (altitude, radius_of_earth, total_isorange_rings, dev_isorange_rings, dev_look_down_angles);

    curandGenerator_t gen;
    unsigned long long seed = 1;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, dev_random_numbers, total_signals);

    // calculate how many signals we need to generate
    blocks = std::ceilf(static_cast<float>(total_signals) / static_cast<float>(threads));

    CalculateSignal << <blocks, threads >> > (
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
        dev_random_numbers,
        dev_signals);

    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, c_start, c_stop);

    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "[CPU] Kernel Execution Took " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
    std::cout << "[GPU] Kernel Execution Took " << ms << " ms" << std::endl;

    start = std::chrono::steady_clock::now();

    cudaErrChk(cudaPeekAtLastError());
    cudaErrChk(cudaDeviceSynchronize());

    cudaErrChk(cudaMemcpy(signals.data(), dev_signals, total_signals * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    cudaErrChk(cudaFree(dev_isorange_rings));
    cudaErrChk(cudaFree(dev_patches));
    cudaErrChk(cudaFree(dev_grazing_angles));
    cudaErrChk(cudaFree(dev_look_down_angles));
    cudaErrChk(cudaFree(dev_signals));

    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "[CPU] Memory Free Took " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

    std::cout << "Complete" << std::endl;
    std::cout << "Write to file? [Y/N]" << std::endl;

    std::string in;
    std::cin >> in;

    if (in == "y" || in == "Y")
    {
        std::cout << "Writing to clutter_signals.txt..." << std::endl;
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
}

int clutter_shared_mem()
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

    float* dev_random_numbers;

    size_t total_az_patches = static_cast<size_t>((azimuth_max - azimuth_min) / delta_azimuth);
    size_t total_signals = (total_isorange_rings * total_az_patches * coherent_pulses);

    std::vector<cuFloatComplex> signals(total_signals);
    cuFloatComplex* dev_signals;

    // start timer
    auto start = std::chrono::steady_clock::now();

    // end of parameters

    // allocate memory on device
    cudaErrChk(cudaMalloc(&dev_signals, total_signals * sizeof(cuFloatComplex)));
    cudaErrChk(cudaMalloc(&dev_random_numbers, total_signals * sizeof(float)));

    int total_mem = total_isorange_rings * sizeof(float);       // isorange rings
    std::cout << "[GPU] Total Memory Usage Isorange Rings: " << total_mem << " bytes" << std::endl;

    total_mem = total_isorange_rings * sizeof(float);          // patch areas
    std::cout << "[GPU] Total Memory Usage Patch Areas: " << total_mem << " bytes" << std::endl;

    total_mem = total_isorange_rings * sizeof(float);          // grazing angles
    std::cout << "[GPU] Total Memory Usage Grazing Angles: " << total_mem << " bytes" << std::endl;

    total_mem = total_isorange_rings * sizeof(float);          // look down angles
    std::cout << "[GPU] Total Memory Usage Look Down Angles: " << total_mem << " bytes" << std::endl;

    //total_mem += total_signals * sizeof(cuFloatComplex);      // signals

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "[GPU] Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;

    std::cout << "[CPU] Memory Allocation Took " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

    start = std::chrono::steady_clock::now();

    // calculate clutter patch areas
    size_t threads = 124;
    size_t blocks = std::ceilf(static_cast<float>(total_isorange_rings) / static_cast<float>(threads));

    // cuda execution time measuring
    cudaEvent_t c_start, c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);

    curandGenerator_t gen;
    unsigned long long seed = 1;

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, dev_random_numbers, total_signals);

    // calculate how many signals we need to generate
    blocks = std::ceilf(static_cast<float>(total_signals) / static_cast<float>(threads));

    CalculateSignalSharedMem<<<blocks, threads>>>(
        total_isorange_rings,
        total_az_patches,
        coherent_pulses,
        altitude,
        radius_of_earth,
        azimuth_min,
        delta_azimuth,
        range_sample_spacing,
        PRF,
        ownship_velocity,
        gamma,
        beam_width,
        transmitted_power,
        wavelength,
        signal_loss,
        dev_random_numbers,
        dev_signals);

    cudaEventRecord(c_stop);
    cudaEventSynchronize(c_stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, c_start, c_stop);

    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "[CPU] Kernel Execution Took " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;
    std::cout << "[GPU] Kernel Execution Took " << ms << " ms" << std::endl;

    start = std::chrono::steady_clock::now();

    cudaErrChk(cudaPeekAtLastError());
    cudaErrChk(cudaDeviceSynchronize());

    cudaErrChk(cudaMemcpy(signals.data(), dev_signals, total_signals * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    cudaErrChk(cudaFree(dev_signals));
    cudaErrChk(cudaFree(dev_random_numbers));

    end = std::chrono::steady_clock::now();
    diff = end - start;

    std::cout << "[CPU] Memory Free Took " << std::chrono::duration<double, std::milli>(diff).count() << " ms" << std::endl;

    std::cout << "Complete" << std::endl;
    std::cout << "Write to file? [Y/N]" << std::endl;

    std::string in;
    std::cin >> in;

    if (in == "y" || in == "Y")
    {
        std::cout << "Writing to clutter_signals.txt..." << std::endl;
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
}

int main()
{
    clutter();
}
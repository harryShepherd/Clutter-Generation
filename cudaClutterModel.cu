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
    float altitude = 500.0f;
    float radius_of_earth = 6.37e6f;
    size_t total_range_rings = 5;
    std::vector<float> isorange_rings_input = { 1000.0f, 2000.0f, 3000.0f, 4000.0f, 5000.0f };
    std::vector<float> grazing_angles = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    float* isorange_rings, * grazing_angle;

    cudaErrChk(cudaMalloc(&isorange_rings, total_range_rings * sizeof(float)));
    cudaErrChk(cudaMalloc(&grazing_angle, total_range_rings * sizeof(float)));

    cudaErrChk(cudaMemcpy(isorange_rings, isorange_rings_input.data(), total_range_rings * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(grazing_angle, grazing_angles.data(), total_range_rings * sizeof(float), cudaMemcpyHostToDevice));

    GrazingAngle<<<1, 256>>>(altitude, radius_of_earth, total_range_rings, isorange_rings, grazing_angle);

    cudaErrChk(cudaPeekAtLastError());
    cudaErrChk(cudaDeviceSynchronize());

    cudaErrChk(cudaMemcpy(grazing_angles.data(), grazing_angle, total_range_rings * sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "Grazing Angle" << std::endl;

    for (float g : grazing_angles)
    {
        std::cout << g << std::endl;
    }
}
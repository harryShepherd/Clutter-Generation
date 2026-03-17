
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

/// <summary>
/// Prints out a summary of device information for all CUDA devices connected to the system.
/// </summary>
void deviceInfo()
{
	int const kb = 1024;
	int const mb = kb * kb;

	cout << "CUDA Version " << CUDART_VERSION << endl;

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	cout << "Device count: " << deviceCount << endl;

	for (int i = 0; i < deviceCount; ++i)
	{
		cudaDeviceProp deviceProperties;
		cudaGetDeviceProperties(&deviceProperties, i);

		cout << i+1 << ": " << deviceProperties.name << " - Compute Capability: " << deviceProperties.major << "." << deviceProperties.minor << endl;
		cout << "\tGlobal Memory: " << deviceProperties.totalGlobalMem / mb << "mb" << endl;
		cout << "\tShared Memory: " << deviceProperties.sharedMemPerBlock / kb << "kb" << endl;
		cout << "\tConstant Memory: " << deviceProperties.totalConstMem / mb << "mb" << endl;
		cout << "\tBlock Registers: " << deviceProperties.regsPerBlock << endl << endl;

		cout << "\tWarp Size: " << deviceProperties.warpSize << endl;
		cout << "\tThread Per Block: " << deviceProperties.maxThreadsPerBlock << endl;
		cout << "\tMax Block Dimensions: [ " << deviceProperties.maxThreadsDim[0] << ", " << deviceProperties.maxThreadsDim[1] << ", " << deviceProperties.maxThreadsDim[2] << " ]" << endl;
		cout << "\tMax Grid Dimensions: [ " << deviceProperties.maxGridSize[0] << ", " << deviceProperties.maxGridSize[1] << ", " << deviceProperties.maxGridSize[2] << " ]" << endl;
	}
}

int total_range_rings = 4;

float CalculateClutterPower(
	float transmitted_power,
	float antenna_gain,
	float clutter_cross_section,
	float wavelength,
	float slant_range,
	float signal_loss
)
{
	float clutter_power = (transmitted_power * powf(antenna_gain, 2.0f) * clutter_cross_section * powf(wavelength, 2.0f)) /
		(powf(4.0f * 3.14159265f , 3.0f) * powf(slant_range, 4.0f) * signal_loss);

	return clutter_power;
}


/// <summary>
/// Calculates the clutter patch areas
/// </summary>
std::vector<float> ClutterPatchArea(
	float altitude,
	float radius_of_earth,
	float delta_azimuth,
	int total_range_rings,
	std::vector<float> isorange_rings
)
{
	std::vector<float> patch_areas(total_range_rings - 1);

	for (int ring = 0; ring < total_range_rings - 1; ++ring)
	{
		patch_areas[ring] = 
			abs(0.5f * delta_azimuth * 
			(powf(isorange_rings[ring], 2.0f) - (powf(isorange_rings[ring + 1], 2.0f))
			* radius_of_earth) / (radius_of_earth + altitude));

	}
	return patch_areas;
}

/// <summary>
/// Calculate the look down angle for each range ring
/// </summary>
void LookDownAngle(
	std::vector<float> isorange_rings,
	float altitude,
	float radius_of_earth,
	int total_range_rings
)
{
	for (int ring = 0; ring < total_range_rings - 1; ++ring)
	{
		float slant_range = (isorange_rings[ring] + isorange_rings[ring + 1]) / 2.0f;

		float look_down_angle = (powf(slant_range, 2.0f) + powf(altitude + radius_of_earth, 2.0f) - powf(radius_of_earth, 2.0f)) / 
			(2.0f * slant_range * (altitude + radius_of_earth));
		
		look_down_angle = asinf(look_down_angle);
	}
}


int main()
{
	deviceInfo();


	float altitude = 50000.0f;
	float radius_of_earth = 4e10f;
	float delta_azimuth = 0.01000000f;
	std::vector<float> isorange_rings = { 1.0f, 2.0f, 3.0f, 4.0f };

	std::vector<float> output = ClutterPatchArea(altitude, radius_of_earth, delta_azimuth, 4, isorange_rings);

	for (int i = 0; i < output.size(); i++)
	{
		cout << output[i] << endl;
	}
}
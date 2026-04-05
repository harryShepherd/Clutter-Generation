
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <random>
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

		// the 0th grid dimension shows a weird number, seems like its overflowing. TODO: find out why
		cout << "\tMax Grid Dimensions: [ " << deviceProperties.maxGridSize[0] << ", " << deviceProperties.maxGridSize[1] << ", " << deviceProperties.maxGridSize[2] << " ]" << endl;
	}
}

int total_range_rings = 4;

float sinc(float x)
{
	if (x == 0.0f)
		return 1.0f;

	return sinf(x) / x;
}

float random_number_0_1()
{
	srand(time(NULL));

	return rand() / RAND_MAX;
}

/// <summary>
/// Simple antenna gain function using a sinc function.
/// </summary>
float AntennaGain(
	float beam_width, // in rads
	float azimuth,
	float elevation
)
{
	float peak_gain = 1.0f / (beam_width * beam_width);

	float az_sinc = sinc(azimuth);
	float el_sinc = sinc(elevation);

	float antenna_gain = peak_gain * az_sinc * az_sinc * el_sinc * el_sinc;

	return antenna_gain;
}

/// <summary>
/// Calculate the power received from a clutter patch.
/// </summary>
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
/// Calculate the grazing angle for each range ring
/// </summary>
std::vector<float> GrazingAngle(
	std::vector<float> isorange_rings,
	float altitude,
	float radius_of_earth,
	int total_range_rings
)
{
	std::vector<float> output(4);

	for (int ring = 0; ring < total_range_rings - 1; ++ring)
	{
		float slant_range = (isorange_rings[ring] + isorange_rings[ring + 1]) / 2.0f;

		float grazing_angle = (powf(slant_range, 2.0f) - powf(altitude + radius_of_earth, 2.0f) + powf(radius_of_earth, 2.0f)) /
			(2.0f * slant_range * (altitude + radius_of_earth));

		if (abs(grazing_angle) < 1)
		{
			grazing_angle = -asinf(grazing_angle);
		}
		else
		{
			grazing_angle = 3.14159263 / 2.0f;
		}

		output[ring] = grazing_angle;
	}

	return output;
}

/// <summary>
/// Calculate the look down angle for each range ring
/// </summary>
std::vector<float> LookDownAngle(
	std::vector<float> isorange_rings,
	float altitude,
	float radius_of_earth,
	int total_range_rings
)
{
	std::vector<float> output(4);

	for (int ring = 0; ring < total_range_rings - 1; ++ring)
	{
		float slant_range = (isorange_rings[ring] + isorange_rings[ring + 1]) / 2.0f;

		float look_down_angle = (powf(slant_range, 2.0f) + powf(altitude + radius_of_earth, 2.0f) - powf(radius_of_earth, 2.0f)) / 
			(2.0f * slant_range * (altitude + radius_of_earth));
		
		if (abs(look_down_angle) < 1)
		{
			look_down_angle = asinf(look_down_angle);
		}
		else
		{
			look_down_angle = 3.14159263 / 2.0f;
		}

		output[ring] = look_down_angle;
	}

	return output;
}

/// <summary>
/// Returns a value that represents the clutter cross section.
/// </summary>
float ClutterCrossSection(
	float scattering_coefficient,
	float grazing_angle
)
{
	return scattering_coefficient * sinf(grazing_angle);
}

/// <summary>
/// Returns power of clutter reflection using a probability density function
/// </summary>
float ClutterReflectivity(
	float reflectivity,
	float cross_section
)
{ 
	return (1.0f / cross_section) * expf(-(reflectivity / cross_section));
}

/// <summary>
/// Calculates the frequency returned by the clutter patch, taking into account doppler shift
/// </summary>
float SignalFrequency(
	float azimuth,
	float elevation,
	float ownship_velocity,
	float wavelength
)
{
	float frequency = -(2.0f * ownship_velocity * cosf(azimuth) * cosf(elevation)) / wavelength;

	return frequency;
}

float ClutterAmplitude(
	float azimuth,
	float elevation,
	float transmitted_power,
	float clutter_cross_section,
	float wavelength,
	float slant_range,
	float signal_loss
)
{
	float antenna_gain = AntennaGain(5.0f, azimuth, elevation);

	return sqrtf(
		CalculateClutterPower(
			transmitted_power,
			antenna_gain,
			clutter_cross_section,
			wavelength,
			slant_range,
			signal_loss
		)
	);
}

std::complex<float> CalculateSignal(
	int range_bin,
	int pulse,
	int PRF,
	float azimuth,
	float elevation,
	float ownship_velocity,
	float transmitted_power,
	float cross_section,
	float wavelength,
	float slant_range,
	float signal_loss

)
{
	// calculate clutter amplitude
	float amplitude = ClutterAmplitude(
		azimuth,
		elevation,
		transmitted_power,
		cross_section,
		wavelength,
		slant_range,
		signal_loss
	);

	// Step 2: calculate doppler frequency
	float doppler_frequency = SignalFrequency(
		azimuth,
		elevation,
		ownship_velocity,
		wavelength
	);

	float transmitted_pulse_time = pulse * 1 / PRF;

	// random starting phase
	float phi_0 = 2.0f * 3.14159265f * random_number_0_1();

	float x = 2.0f * 3.14159265f * doppler_frequency * transmitted_pulse_time + phi_0;

	std::complex<float> signal_c(amplitude * cosf(x), -amplitude * sinf(x));

	return signal_c;
}

int main()
{
	deviceInfo();


	float altitude = 5000.0f;
	float radius_of_earth = 10.0f;
	float delta_azimuth = 0.01000000f;
	std::vector<float> isorange_rings = { 1.0f, 2.0f, 3.0f, 4.0f };

	std::vector<float> output = ClutterPatchArea(altitude, radius_of_earth, delta_azimuth, 4, isorange_rings);
	std::vector<float> output2 = LookDownAngle(isorange_rings, altitude, radius_of_earth, 4);

	for (int i = 0; i < output.size(); i++)
	{
		cout << output[i] << endl;
	}

	cout << endl;

	for (int i = 0; i < output.size(); i++)
	{
		cout << output2[i] << endl;
	}
}
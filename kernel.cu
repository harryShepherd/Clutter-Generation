#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

using namespace std;

// Prints out a summary of device information for all CUDA devices connected to the system.
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
		cout << endl;
	}
}

// Used for calculations
int total_range_rings = 4;

// Sinc Function
float sinc(float x)
{
	if (x == 0.0f)
		return 1.0f;

	return sinf(x) / x;
}

// Generates random float between 0 and 1
float random_number_0_1()
{
	srand(time(NULL));

	return rand() / RAND_MAX;
}

// Simple antenna gain function using sinc to roughly imitate boresight and sidelobes
float AntennaGain(
	float beam_width,
	float azimuth,
	float elevation
)
{
	float az_sinc = sinc(azimuth);
	float el_sinc = sinc(elevation);

	float antenna_gain = beam_width * az_sinc * az_sinc * el_sinc * el_sinc;

	return antenna_gain;
}

// Calculate the power received from a clutter patch.
float CalculateClutterPower(
	float transmitted_power,
	float antenna_gain,
	float clutter_cross_section,
	float wavelength,
	float slant_range,
	float signal_loss
)
{
	// The general clutter power equation
	float clutter_power = 
		(transmitted_power * powf(antenna_gain, 2.0f) * clutter_cross_section * powf(wavelength, 2.0f)) /
		(powf(4.0f * 3.14159265f , 3.0f) * powf(slant_range, 4.0f) * signal_loss);

	return clutter_power;
}

// Calculates the clutter patch areas
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
			(powf(isorange_rings[ring], 2.0f) - (powf(isorange_rings[ring + 1], 2.0f)) * 
			radius_of_earth) / (radius_of_earth + altitude));

	}
	return patch_areas;
}

// Calculate the grazing angle for each range ring
std::vector<float> GrazingAngle(
	std::vector<float> isorange_rings,
	float altitude,
	float radius_of_earth,
	int total_range_rings
)
{
	std::vector<float> output(4);

	// For every range ring
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

// Calculate the look down angle for each range ring
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

// Returns a value that represents the clutter cross section.
float ClutterCrossSection(
	float scattering_coefficient,
	float grazing_angle
)
{
	return scattering_coefficient * sinf(grazing_angle);
}

// Returns power of clutter reflection using a probability density function
float ClutterReflectivity(
	float reflectivity,
	float cross_section
)
{ 
	return (1.0f / cross_section) * expf(-(reflectivity / cross_section));
}

// Calculates the relative amplitude of a returned signal
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

// Calculates the frequency returned by the clutter patch, taking into account doppler shift
float Frequency(
	float azimuth,
	float elevation,
	float ownship_velocity,
	float wavelength
)
{
	float frequency = -2.0f * ownship_velocity * cosf(azimuth) * cosf(elevation) / wavelength;

	return frequency;
}

// Calculate the signal 
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
	float doppler_frequency = Frequency(
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

	int prf_count = 10000;
	int pulse_count = 10000;

	float transmitted_pulse_time = pulse_count * (1 / prf_count);

	for (int sample = 1; sample <= pulse_count; ++sample)
	{
		// Signal is calculated here
		// sampled at each range for the number of coherent pulsese transmitted at the PRF
	}
}
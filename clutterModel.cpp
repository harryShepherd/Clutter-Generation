#define _USE_MATH_DEFINES

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include <fstream>

using namespace std;

// Sinc Function
float sinc(float x)
{
	if (x == 0.0f)
		return 1.0f;

	return sinf(x) / x;
}

std::random_device rd;
std::mt19937 generator(rd());
std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

// Generates random float between 0 and 1
float random_number_0_1()
{
    return distribution(generator);
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
	std::vector<float> patch_areas(total_range_rings);

	for (int ring = 0; ring < total_range_rings; ++ring)
	{
        float r1 = isorange_rings[ring];
        float r2 = isorange_rings[ring + 1];

        float area =
            0.5f *
            delta_azimuth *
            (r2 * r2 - r1 * r1) *
            radius_of_earth /
            (radius_of_earth + altitude);

        patch_areas[ring] = area;

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
	std::vector<float> output(total_range_rings);

	// For every range ring
	for (int ring = 0; ring < total_range_rings; ++ring)
	{
		float slant_range = (isorange_rings[ring] + isorange_rings[ring + 1]) / 2.0f;

		float grazing_angle = (powf(slant_range, 2.0f) - powf(altitude + radius_of_earth, 2.0f) + powf(radius_of_earth, 2.0f)) /
			(2.0f * slant_range * radius_of_earth);

		if (abs(grazing_angle) < 1)
		{
			grazing_angle = -(asinf(grazing_angle));
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
	std::vector<float> output(total_range_rings);

	for (int ring = 0; ring < total_range_rings; ++ring)
	{
		float slant_range = (isorange_rings[ring] + isorange_rings[ring + 1]) / 2.0f;

        if (slant_range < altitude)
        {
            output[ring] = NAN;
            continue;
        }

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

float MeanClutterCrossSection(
	float scattering_coefficient,
	float grazing_angle
)
{
	return scattering_coefficient * sinf(grazing_angle);
}

float GenerateRandomSigma0(float mean_sigma0)
{
    float u = random_number_0_1();

    return -mean_sigma0 * std::logf(u);
}

float ClutterCrossSection(
    float gamma,
    float grazing_angle,
    float patch_area
)
{
    float mean_sigma0 = MeanClutterCrossSection(gamma, grazing_angle);
    float sigma0 = GenerateRandomSigma0(mean_sigma0);
    float cross_section = sigma0 * patch_area;
    
    return cross_section;
}

// Calculates the frequency returned by the clutter patch, taking into account doppler shift
float Frequency(
	float azimuth,
	float elevation,
	float ownship_velocity,
	float wavelength
)
{
	float frequency = (-2.0f * ownship_velocity * cosf(azimuth) * cosf(elevation)) / wavelength;

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
    float patch_power,
	float cross_section,
	float wavelength,
	float slant_range,
	float signal_loss,
    float phi_0 // generated each patch
)
{
	// calculate clutter amplitude
	float amplitude = sqrtf(patch_power);

	// Step 2: calculate doppler frequency
	float doppler_frequency = Frequency(
		azimuth,
		elevation,
		ownship_velocity,
		wavelength
	);

	float transmitted_pulse_time = pulse * (1 / (float)PRF);

	float x = 2.0f * 3.14159265f * doppler_frequency * transmitted_pulse_time + phi_0;

	std::complex<float> signal_c(amplitude * cosf(x), -amplitude * sinf(x));

	return signal_c;
}

int main()
{

    // parameters

    const float transmitted_power = 15000.0f;  // W
    const float wavelength = 0.03f;            // m
    const float PRF = 15000.0f;                // Hz

    const float beam_width = 40.0f;

    const int coherent_pulses = 64;
    const int range_samples = 100;

    const float range_sample_spacing = 100.0f; // m

    const float altitude = 5250.0f;             // m
    const float ownship_velocity = 500.0f;      // m/s

    const float radius_of_earth = 6.371e6f;     // m

    const float look_down_boresight = 45.0f * M_PI / 180.0f;

    const float look_aside_boresight = 1.0f * M_PI / 180.0f;

    const float delta_azimuth = 0.1f * M_PI / 180.0f;

    float azimuth_min = -5.0f * M_PI / 180.0f; // -5 degs
    float azimuth_max =  5.0f * M_PI / 180.0f; // +5 degs

    const float gamma = 0.1f;

    const float signal_loss = 1.0f; // 1 means no signal loss

    std::vector<float> isorange_rings;
    int total_isorange_rings = 200;

    for (int i = 0; i <= total_isorange_rings; ++i)
    {
        isorange_rings.push_back(i * range_sample_spacing);
    }

    // end of parameters

    std::vector<float> patches = ClutterPatchArea(
        altitude, 
        radius_of_earth,
        delta_azimuth, 
        total_isorange_rings, 
        isorange_rings
    );

    std::vector<float> grazing_angles = GrazingAngle(
        isorange_rings,
        altitude,
        radius_of_earth,
        total_isorange_rings
    );

    std::vector<float> look_down_angles = LookDownAngle(
        isorange_rings, 
        altitude, 
        radius_of_earth, 
        total_isorange_rings
    );

    std::vector<std::complex<float>> signals;

    std::ofstream output_file("clutter_signals.txt");

    if (!output_file)
    {
        std::cerr << "Failed to open output file.\n";
        return 1;
    }

    // iterating through isorange rings
    for(int ring = 0; ring < patches.size(); ++ring)
    {
        int azimuth_patches = static_cast<int>((azimuth_max - azimuth_min) / delta_azimuth);

        for (int az = 0; az < azimuth_patches; ++az)
        {
            float patch_area = patches[ring];
            float patch_elevation = look_down_angles[ring];
            float grazing_angle = grazing_angles[ring];

            float slant_range = (isorange_rings[ring] + isorange_rings[ring + 1]) / 2.0f;

            float cross_section = ClutterCrossSection(
                gamma,
                grazing_angle,
                patch_area
            );

            float azimuth = azimuth_min + az * delta_azimuth;

            float patch_azimuth = azimuth + (delta_azimuth / 2.0f);

            float azimuth_offset = patch_azimuth - look_aside_boresight;
            float elevation_offset = patch_elevation - look_down_boresight;

            float antenna_gain = AntennaGain(beam_width, azimuth_offset, elevation_offset);

            float power = CalculateClutterPower(
                transmitted_power,
                antenna_gain,
                cross_section,
                wavelength,
                slant_range,
                signal_loss
            );

            // random phase for entire patch
            float phi_0 = 2.0f * M_PI * random_number_0_1();

            for(int pulse = 0; pulse < coherent_pulses; ++pulse)
            {
                std::complex<float> generated_signal = CalculateSignal(
                    isorange_rings[ring],
                    pulse,
                    PRF,
                    patch_azimuth,
                    patch_elevation,
                    ownship_velocity,
                    transmitted_power,
                    power,
                    cross_section,
                    wavelength,
                    slant_range,
                    signal_loss,
                    phi_0);

                if (std::isnan(generated_signal.imag()) || std::isnan(generated_signal.real()))
                {
                    continue;
                }

                signals.push_back(generated_signal);

                output_file << ring << " "
                << az << " "
                << pulse << " "
                << generated_signal.real() << " "
                << generated_signal.imag() << "\n";
            }
        }
    }

    std::cout << "Total signals: " << signals.size() << std::endl;
}

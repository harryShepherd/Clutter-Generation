#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "frequecy.cu"

__device__ cuDoubleComplex CalculateSignal(
    const int range_bin,
    const int pulse,
    const int prf,
    const float az,
    const float el,
    const float vel,
    const float trans_power,
    const float patch_power,
    const float wavelength,
    const float slant_range
    const float signal_loss,
    const float phi_0
)
{
    float amplitude = sqrtf(patch_power);

    float doppler_frequency = Frequency(
        az,
        el,
        vel,
        wavelength
    );

    float transmitted_pulse_time = pulse * (1.0f / (float)prf);

    float x = 2.0f * CUDART_PI_F * doppler_frequency * transmitted_pulse_time + phi_0;

    float sinx, cosx;
    sincosf(x, &sinx, &cosx);

    cuComplexDouble c = make_cuDoubleComplex(amplitude * cosx, -amplitude * sinx);

    return c;
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
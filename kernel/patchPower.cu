#include "patchPower.cuh"

__device__ float patchPower(
	float transmitted_power,
	float antenna_gain,
	float cross_section,
	float wavelength,
	float slant_range,
	float signal_loss
)
{
	// The general clutter power equatio
	float clutter_power =
		(transmitted_power * powf(antenna_gain, 2.0f) * cross_section * powf(wavelength, 2.0f)) /
		(powf(4.0f * CUDART_PI, 3.0f) * powf(slant_range, 4.0f) * signal_loss);

}
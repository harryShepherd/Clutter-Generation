#include "signal.cuh"
#include "frequency.cuh"

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
    const float slant_range,
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

    float sinx = sinf(x);
    float cosx = cosf(x);

    cuDoubleComplex c = make_cuDoubleComplex(amplitude * cosx, -amplitude * sinx);

    return c;
}
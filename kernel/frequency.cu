#include "frequency.cuh"

__device__ float Frequency(
    const float az,
    const float el,
    const float vel,
    const float wavelength
)
{
    float frequency = (-2.0f * vel * cosf(az) * cosf(el)) / wavelength;

    return frequency;
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ inline float Frequency(
    const float az,
    const float el,
    const float vel,
    const float wavelength
)
{
    float frequency = (-2.0f * ownship_velocity * cosf(az) * cosf(el)) / wavelength;

    return frequency;
}
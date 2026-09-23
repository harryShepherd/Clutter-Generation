#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "maths_constants.h"
#include "utils/sinc.cu"

__device__ float AntennaGain(
    float beam_width,
    float azimuth,
    float elevation)
{
    float az_sinc = sincf(azimuth / beam_width);
    float el_sinc = sincf(elevation / beam_width);

    float gain = pow(az_sinc, 2) * pow(el_sinc, 2);
    
    return gain;
}
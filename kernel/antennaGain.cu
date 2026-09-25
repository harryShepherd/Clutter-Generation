#include "utils/sinc.cu"

__device__ float AntennaGain(
    float beam_width,
    float azimuth,
    float elevation)
{
    float az_sinc = sincf(azimuth / beam_width);
    float el_sinc = sincf(elevation / beam_width);

    float gain = powf(az_sinc, 2) * powf(el_sinc, 2);
    
    return gain;
}
#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "math_constants.h"
#include <cmath>

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
);
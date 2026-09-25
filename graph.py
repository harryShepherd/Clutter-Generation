import numpy as np
import matplotlib.pyplot as plt

filename = "clutter_signals.txt"

# dimensions
number_of_range_patches = 200
number_of_azimuth_patches = 100
number_of_pulses = 64

PRF = 15000.0
range_spacing = 100.0

# read file
data = np.loadtxt(filename)

real = data[:, 0]
imaginary = data[:, 1]

complex_signal = real + 1j * imaginary


# check number of signals
expected_signals = (
    number_of_range_patches
    * number_of_azimuth_patches
    * number_of_pulses
)

if len(complex_signal) != expected_signals:
    raise ValueError(
        f"Expected {expected_signals} signals, "
        f"but found {len(complex_signal)}."
    )


print("Total signals:", len(complex_signal))

# reshape the complex signal along our dimensions
signals = complex_signal.reshape(
    number_of_range_patches,
    number_of_azimuth_patches,
    number_of_pulses
)


# sum all azimuth patches
range_pulse = np.sum(
    signals,
    axis=1
)


# perform fft & fft shit
doppler_data = np.fft.fft(
    range_pulse,
    axis=1
)

doppler_data = np.fft.fftshift(
    doppler_data,
    axes=1
)


# calculate power in decibels
power = np.abs(doppler_data) ** 2

power_db = 10 * np.log10(
    power / np.max(power) + 1e-12
)

# get the doppler frequencies
doppler_frequencies = np.fft.fftshift(
    np.fft.fftfreq(
        number_of_pulses,
        d=1.0 / PRF
    )
)

# get ranges
ranges = (
    np.arange(number_of_range_patches) + 0.5
) * range_spacing

# plot the graph
plt.figure(figsize=(10, 7))

plt.imshow(
    power_db.T,
    aspect="auto",
    origin="upper",
    extent=[
        ranges[0],
        ranges[-1],
        doppler_frequencies[-1],
        doppler_frequencies[0]
    ]
)

plt.xlabel("Range (m)")
plt.ylabel("Doppler frequency (Hz)")
plt.colorbar(label="Relative power (dB)")

plt.show()
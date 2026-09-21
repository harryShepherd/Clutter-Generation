import numpy as np
import matplotlib.pyplot as plt

filename = "clutter_signals.txt"

# range_patch  azimuth_patch  pulse  I Q
data = np.loadtxt(filename, skiprows=1)

range_patch = data[:, 0].astype(int)
azimuth_patch = data[:, 1].astype(int)
pulse = data[:, 2].astype(int)

real = data[:, 3]
imaginary = data[:, 4]


complex_signal = real + 1j * imaginary


number_of_range_patches = range_patch.max() + 1
number_of_pulses = pulse.max() + 1

print("Range patches:", number_of_range_patches)
print("Pulses:", number_of_pulses)


# Each range/pulse cell will contain the sum of all
# azimuth-patch contributions.

range_pulse = np.zeros(
    (number_of_range_patches, number_of_pulses),
    dtype=np.complex128
)

np.add.at(
    range_pulse,
    (range_patch, pulse),
    complex_signal
)


# FFT is performed across the coherent pulses.
doppler_data = np.fft.fft(
    range_pulse,
    axis=1
)

# Move zero Doppler to the centre.
doppler_data = np.fft.fftshift(
    doppler_data,
    axes=1
)


power = np.abs(doppler_data) ** 2

# Convert to relative dB.
power_db = 10 * np.log10(
    power / np.max(power) + 1e-12
)


PRF = 15000.0          # Hz
range_spacing = 100.0 # metres

# Doppler frequencies corresponding to the FFT bins.
doppler_frequencies = np.fft.fftshift(
    np.fft.fftfreq(
        number_of_pulses,
        d=1.0 / PRF
    )
)

# Centre of each range cell.
ranges = (
    np.arange(number_of_range_patches) + 0.5
) * range_spacing

X, Y = np.meshgrid(
    np.arange(number_of_range_patches),
    np.arange(number_of_pulses)
)

# Contour levels in relative dB.
levels = np.arange(
    -60,
    -4,
    5
)

plt.figure(
    figsize=(10, 7)
)

plt.contour(
    X,
    Y,
    power_db.T,
    levels=levels,
    linewidths=0.5
)


plt.imshow(power_db.T, aspect="auto", origin="lower")
plt.xlabel("Range")
plt.ylabel("Doppler")
plt.show()
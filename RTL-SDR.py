from rtlsdr import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.io import wavfile
import numpy as np

# sdr = RtlSdr()
#
# # configure device
# sdr.sample_rate = 2.4e6
# sdr.center_freq = 923.8375e6
# print(sdr.get_gains())
# sdr.gain = 77
#
# samples = sdr.read_samples(4096)
# samples = sdr.read_samples(4800000)
#
# mag = np.sqrt(np.square(np.real(samples)) + np.square(np.imag(samples)))
# plt.plot(np.real(samples[:1000]))
# print(np.real(samples[:10]))
# print(mag[:10])
# # plt.plot(20 * np.log10(mag))
# plt.show()

# time_step = 1 / sdr.sample_rate
# # The FFT of the signal
# sig_fft = scipy.fftpack.fft(samples)
# # print(sig_fft)
#
# # And the power (sig_fft is of complex dtype)
# power = np.abs(sig_fft)
# power = scipy.fftpack.fftshift(power)
#
# # The corresponding frequencies
# sample_freq = scipy.fftpack.fftfreq(len(sig_fft)) * sdr.sample_rate
# sample_freq = np.sort(sample_freq)
# # sample_freq += 87.6e6
#
# # Plot the FFT power
# plt.figure(figsize=(6, 5))
# plt.plot(sample_freq, power)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('power')
# plt.show()

# sdr.close()


def fm_demod(x, df=1.0, fc=0.0):
    ''' Perform FM demodulation of complex carrier.

    Args:
        x (array):  FM modulated complex carrier.
        df (float): Normalized frequency deviation [Hz/V].
        fc (float): Normalized carrier frequency.

    Returns:
        Array of real modulating signal.
    '''

    # Remove carrier.
    n = np.arange(len(x))
    rx = x*np.exp(-1j*2*np.pi*fc*n)

    # Extract phase of carrier.
    phi = np.arctan2(np.imag(rx), np.real(rx))

    # Calculate frequency from phase.
    y = np.diff(np.unwrap(phi)/(2*np.pi*df))

    return y


i_samples = np.zeros(100000)
q_samples = np.zeros(100000)
data_samples = np.zeros(100000)

f = open("E:\Projects\SDR\sdrsharp-x86\FM106.3_107100000Hz_2.4Msps_IQ.wav", 'rb')
print(f.read(4))
print(int.from_bytes(f.read(4), byteorder='little'))
print(f.read(4))
print(f.read(4))
print(int.from_bytes(f.read(4), byteorder='little'))
print(int.from_bytes(f.read(2), byteorder='little'))
print(int.from_bytes(f.read(2), byteorder='little'))
print(int.from_bytes(f.read(4), byteorder='little'))
print(int.from_bytes(f.read(4), byteorder='little'))
print(int.from_bytes(f.read(2), byteorder='little'))
print(int.from_bytes(f.read(2), byteorder='little'))
print(f.read(4))
print(int.from_bytes(f.read(4), byteorder='little'))
for i in range(100000):
    i_samples[i] = int.from_bytes(f.read(1), byteorder='little', signed=True)
    q_samples[i] = int.from_bytes(f.read(1), byteorder='little', signed=True)
data_samples = i_samples + 1j * q_samples

aud = fm_demod(data_samples)
plt.plot(aud)
plt.show()

# plt.plot(np.sqrt(np.square(i_samples) + np.square(q_samples)))
# plt.show()

# fs, data = wavfile.read("E:\Projects\SDR\sdrsharp-x86\FM106.3_107100000Hz_2.4Msps_IQ.wav")
# data = data.astype(float)
# I, Q = data[:500000, 0], data[:500000, 1]
# A = np.sqrt(I*I + Q*Q)
#
# plt.plot(A)
# plt.show()





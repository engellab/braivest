### NOTE: This script is not runnable, it is just an example of how to use the data after you have loaded your own data.

import os
import numpy as np
from braivest.model.wavelets import calculate_wavelet_coeffs, calculate_wavelet_power, get_wavelet_freqs, pywt_frequency2scale

path_prefix = 'path/to/your/data'  # Change this to your data path
file_name = 'example_data'  # Change this to your file name
sample_rate = 1000  # Change this to your sample rate
channel = 0  # Change this to the channel you want to analyze, e.g., 0 for the first channel


data = np.empty() #Load your own data. Assume data is of shape (n_samples, n_channels)
freqs = get_wavelet_freqs(0.5, 50, 50) # start, stop, number of frequencies. can play around with this
scales = pywt_frequency2scale('cmor1.5-1.0', freqs, sample_rate)
subsample=10
wavelet_coeffs, freqs = calculate_wavelet_coeffs(data[:, channel], wavelet_name='cmor1.5-1.0', scales=scales,sampling_rate=sample_rate, highpass=0.3, z_score=True)
wavelet_power = calculate_wavelet_power(wavelet_coeffs, subsample=subsample)

# Save the wavelet power
if not os.path.exists(os.path.join(path_prefix, file_name, 'wavelets')):
    os.makedirs(os.path.join(path_prefix, file_name, 'wavelets'))
save_path = os.path.join(path_prefix, file_name, 'wavelets/wavelet_power_channel_{}.npy'.format(channel))
np.save(save_path, wavelet_power)

import pywt 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy.stats import zscore

def butter_filter(data, lowcut=None, highcut=None, fs=1, order=5, axis=-1):
	"""
	Butterworth filter some data
	Inputs: 
		data (dtype: ndarray): the input signal as 1-D array 
		lowcut (dtype: float): lowcut frequency
		highcut (dtype: float): highcut frequency
		fs (dtype: float): the sample rate of the input signal
		order (dtype: int, default=5): the order of the filter
	Returns:
		- filtered data (dtype: ndarray)
	"""
	b, a = butter_pass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data, axis=axis)
	return y

def butter_pass(lowcut=None, highcut=None, fs=1, order=5):
	"""
	Butterworth filter some data
	Inputs: 
		lowcut (dtype: float): lowcut frequency
		highcut (dtype: float): highcut frequency
		fs (dtype: float): the sample rate of the input signal
		order (dtype: int, default=5): the order of the filter
	Returns:
		- b, a coefficients (dtype: tuple)
	"""
	nyq = 0.5 * fs
	if lowcut is not None and highcut is not None:
		normal_lowcut = lowcut / nyq
		normal_highcut = highcut / nyq
		b,a = butter(order,(normal_lowcut, normal_highcut), btype='bandpass', analog=False)
		return b,a
	elif lowcut is not None:
		nyq = 0.5 * fs
		normal_cutoff = lowcut / nyq
		b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)
		return b, a
	elif highcut is not None:
		nyq = 0.5 * fs
		normal_cutoff = highcut / nyq
		b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
		return b, a
	else:
		raise ValueError("Either lowcut or highcut must be provided")


def butter_highpass(cutoff, fs, order=5):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
	return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
	"""
	Lowpass filter some data
	Inputs: 

	"""
	b, a = butter_highpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def get_wavelet_freqs(start_freq, end_freq, n_bands):
	"""
	Inputs:
	Find frequencies in log space from given start and end frequencies.
	start_freq (dtype: float): starting frequency
	end_freq (dtype: float): ending frequency
	n_bands (dtype: int): number of frequencies

	Returns:
	- computed frequencies
	"""
	freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), n_bands)
	return freqs

def pywt_frequency2scale(wavelet,frequencies,sampling_rate):
	"""
	Pywt helper function Given wavelet wavelet_name, range of frequencies along with sampling rate for a given signal, 
	returns the scales that would capture these frequencies. 
	Input:
		wavelet: (dtype: string), wavelet name: example 'cmor1.5-1.0'
		frequencies: (dtype: list of floats), desired frequencies for the wavelets
		sampling_rate: (dtype: float), sampling rate of raw data
	Returns:
		The scales that capture the frequency (dtype: list of floats)
	"""

	# There is a linear relationship (inverse times the some constant) between scale and frequency
	example_scale = 2
	example_frequency = pywt.scale2frequency(wavelet=wavelet,scale=example_scale)*sampling_rate
	
	scales = []
	
	for frequency in frequencies:
		scale  = (example_scale*example_frequency)/frequency # a simple proportions calculation
		scales.append(scale)
	return scales

def calculate_wavelet_coeffs(recording, wavelet_name, scales, sampling_rate, highpass=0, z_score=True, nan_policy='remove'):
	""" 
	Calculate the wavelet coefficients of a signal. See pywt for more reference.
	Input: 
		recording: (dtype: ndarray) the input signal as 1-D array 
		wavelet_name: (dtype: string) the name the wavelet
		sampling_rate: (dtype: float) the sample rate of the input signal
		highpass: (dtype: float, default=0), a frequency threshold to highpass the data
		zscore: (dtype: bool, default=True) whether or not to zscore the input signal before calculating wavelets
	Returns:
		- calculated wavelet coefficients (dtype: ndarray)
		- the frequencies of the calculated wavelets (dtype: ndarray)
	"""
	if nan_policy =='remove':
		recording[np.isnan(recording)] = np.nanmax(recording)
		if highpass > 0:
			recording = butter_highpass_filter(recording, highpass, sampling_rate)
		if z_score:
			recording = zscore(recording, nan_policy='omit')
		[coefficients, frequencies] = pywt.cwt(recording, scales, wavelet_name, 1.0/sampling_rate)
	elif nan_policy=='split':
		if z_score:
			recording = zscore(recording, nan_policy='omit')
		recording_splits = np.split(recording, np.where(np.isnan(recording))[0])
		del recording
		coefficients = []
		for split in recording_splits:
			if split[0] == np.nan:
				split_coefficients = np.empty((len(scales), 1))
				split_coefficients[:] = np.nan
				coefficients.append(split_coefficients)
			if len(split) > 1:
				if split[0] == np.nan:
					split = split[1:]
				if highpass > 0:
					split = butter_highpass_filter(split, highpass, sampling_rate)
				[split_coefficients, frequencies] = pywt.cwt(split, scales, wavelet_name, 1.0/sampling_rate)
				coefficients.append(split_coefficients)
		coefficients = np.concatenate(coefficients, axis=1)
	return coefficients.T, frequencies #want data to be of shape (nsamples, scales)

def calculate_wavelet_power(coefficients, subsample= 1, nan_policy='remove'):
	"""
	Calculate wavelet power from wavelet coefficients.
	Input:
		 coefficient (dtype: ndarray): wavelet coefficients calculated using calculate_wavelet_coeffs with shape (nsamples, scales)
		 subsample (dtype: int, default=1): Factor to subsample the coefficients in time
	Returns:
		The wavelet power (dtype: ndarray)
	"""
	power = np.log2(np.square(np.abs(coefficients)))
	#replace NaNs with medium value
	if nan_policy =='remove':
		power[np.isnan(power)] =  np.mean(power[ ~ np.isnan(power)])
	power = power[::subsample, :]
	return power
	

def plot_scales(frequencies, sampling_rate, scales, wavelet_name, fig_width=12,fig_height=18,common_scale=True,columns=1,vpadding=0.2,hpadding=0.2):
		"""
		Returns a figure showing requested scales 

		Source: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html (bottom plot) 
		"""
		freqs = frequencies
		sampling_period = 1/sampling_rate

		wav = pywt.ContinuousWavelet(wavelet_name)
		# the range over which the wavelet will be evaluated
		width = wav.upper_bound - wav.lower_bound
		max_len = int(np.max(scales)*width + 1)
		t = np.arange(max_len)

		intensities = []
		fig = plt.figure(figsize=(fig_width,fig_height))
		for n, scale in enumerate(scales):
			
			axs_a =  plt.subplot(len(scales), columns*2,n*2+1)
			axs_b  =  plt.subplot(len(scales), columns*2,n*2+1+1)

			# The following code is adapted from the internals of cwt
			int_psi, x = pywt.integrate_wavelet(wav, precision=10)
			step = x[1] - x[0]
			j = np.floor(
				np.arange(scale * width + 1) / (scale * step))
			if np.max(j) >= np.size(int_psi):
				j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
			j = j.astype(np.int)

			# normalize int_psi for easier plotting
			int_psi /= np.abs(int_psi).max()

			# discrete samples of the integrated wavelet
			filt = int_psi[j][::-1]

			# The CWT consists of convolution of filt with the signal at this scale
			# Here we plot this discrete convolution kernel at each scale.

			nt = len(filt)
			t = np.linspace(-nt//2, nt//2, nt)
			axs_a.plot(t, filt.real, t, filt.imag)
			if(common_scale):
				axs_a.set_xlim([-max_len//2, max_len//2])
			axs_a.set_ylim([-1, 1.2])
			#axs_a.text(50, 0.35, 'scale = {}, freqs={}'.format(round(scale,2),round(freqs[n],2) ))
			axs_a.text(50, 0.35, 'freqs={}Hz'.format(round(freqs[n],2) ))

			f = np.linspace(-np.pi, np.pi, max_len) / (2*np.pi*sampling_period)
			f = -f[0:max_len//2]
			f = np.flip(f)
			
			freq_range_indexes = np.where( (f >= np.min(freqs)) & (f<= np.max(freqs) ) ) # indices between min/max frequencies,
			if(common_scale): # for visualization 
				selected_indices =  np.where(f< np.max(freqs) + 0.5*np.max(freqs)) # max cap
			else: # separate window for each scale
				selected_indices =  np.where( (f > freqs[n] *0.5) &  (f <  freqs[n] *1.5)  )

			frequency_range = f[freq_range_indexes]

			f = f[ selected_indices  ]

			filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
			filt_fft /= np.abs(filt_fft).max()
			filt_fft = -filt_fft[0:max_len//2]
			filt_fft = np.flip(filt_fft)

			intensities.append( np.abs(filt_fft[freq_range_indexes] )**2)
			filt_fft = filt_fft[selected_indices]
			#filt_fft = filt_fft[0:int(np.max(freqs))+10]
			
			axs_b.plot(f, np.abs(filt_fft)**2 )
		
			axs_b.set_ylim([0, 1])
			axs_b.grid(True, axis='x')

			if(n in np.arange(0,columns )):
				axs_a.set_title('Wavelet {}'.format(wavelet_name))
				title = r'|FFT(filter)|$^2$'
				axs_b.set_title(title)
				axs_a.legend(['real', 'imaginary'], loc='upper left')
				axs_b.legend(['Power'], loc='upper left')
			
			if(n in np.arange(len(scales)-columns, len(scales) )):
				axs_a.set_xlabel('time (samples)')
				axs_b.set_xlabel('frequency (Hz)')

		left = 0.125  # the left side of the subplots of the figure
		right = 0.9   # the right side of the subplots of the figure
		bottom = 0.1  # the bottom of the subplots of the figure
		top = 0.9     # the top of the subplots of the figure
		wspace = hpadding  # the amount of width reserved for space between subplots,
					# expressed as a fraction of the average axis width
		hspace = vpadding  # the amount of height reserved for space between subplots,
					# expressed as a fraction of the average axis height
		plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

		plt.close(fig)
		return(fig)

def plot_coverage(frequencies, sampling_rate, scales, wavelet_name, fig_width=12,fig_height=18,common_scale=True,columns=1,vpadding=0.2,hpadding=0.2):
		"""
		Returns a figure showing the coverage of wavelets over selected frequencies.
		Provided a rough estimate of coverage and redundancy percentage (normalized area in the range 0-1).

		Source: https://pywavelets.readthedocs.io/en/latest/ref/cwt.html (bottom plot) 
		"""
		freqs = frequencies
		sampling_period = 1/sampling_rate

		wav = pywt.ContinuousWavelet(wavelet_name)
		# the range over which the wavelet will be evaluated
		width = wav.upper_bound - wav.lower_bound
		max_len = int(np.max(scales)*width + 1)
		t = np.arange(max_len)

		intensities = []
		fig = plt.figure(figsize=(fig_width,fig_height))
		axs_a = plt.subplot(2,1,1)
		axs_b = plt.subplot(2,1,2)


		for n, scale in enumerate(scales):
			
			# The following code is adapted from the internals of cwt
			int_psi, x = pywt.integrate_wavelet(wav, precision=10)
			step = x[1] - x[0]
			j = np.floor(
				np.arange(scale * width + 1) / (scale * step))
			if np.max(j) >= np.size(int_psi):
				j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
			j = j.astype(np.int)

			# normalize int_psi for easier plotting
			int_psi /= np.abs(int_psi).max()

			# discrete samples of the integrated wavelet
			filt = int_psi[j][::-1]

			# The CWT consists of convolution of filt with the signal at this scale
			# Here we plot this discrete convolution kernel at each scale.

			nt = len(filt)
			t = np.linspace(-nt//2, nt//2, nt)

			f = np.linspace(-np.pi, np.pi, max_len) / (2*np.pi*sampling_period)
			f = -f[0:max_len//2]
			f = np.flip(f)
			
			freq_range_indexes = np.where( (f >= np.min(freqs)) & (f<= (np.max(freqs)*1.5) ) ) # indices between min/max frequencies, with some leverage - for plotting
			frequency_range = f[freq_range_indexes]
			interest_freq_range_indexes = (np.where( (frequency_range >= np.min(freqs)) & ( frequency_range<= np.max(freqs) ) ) )[0]   # indices between min/max frequencies within EXACT min max range of interest: for calculations


			filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
			filt_fft /= np.abs(filt_fft).max()
			filt_fft = -filt_fft[0:max_len//2]
			filt_fft = np.flip(filt_fft)

			intensities.append( np.abs(filt_fft[freq_range_indexes] )**2)


		# coverage

		for intensity in intensities:
			axs_b.plot(frequency_range,intensity)


		intensities = np.sum(np.asarray(intensities),axis=0)
		intensities_max = np.copy(intensities)
		intensities_min = np.copy(intensities)
		intensities_min[intensities_min<1]=1
		intensities_max[intensities_max>1]=1


		# Plot #1 - summed coverage
		axs_a.plot(frequency_range, intensities, color="black"  )
		axs_a.fill_between(frequency_range, 1, intensities_min, color='blue', alpha=0.3, interpolate=False)
		axs_a.fill_between(frequency_range, intensities_max,1, color='red', alpha=0.3, interpolate=False)
		axs_a.fill_between(frequency_range, 0, intensities_max, color='green', alpha=0.3, interpolate=False) #captured

		# "IDEAL" box
		axs_a.hlines(xmin=min(frequency_range),xmax=max(freqs),y=1)
		axs_a.vlines(ymax=1,ymin=0, x= np.max(freqs ) )
		axs_a.vlines(ymax=1,ymin=0, x= np.min(freqs ) )

		# Formatting
		axs_a.set_ylim(0, 1.3*np.max(intensities))
		axs_b.set_ylim(0, 1.1)
		axs_a.set_xlabel("Frequency [Hs]")
		axs_b.set_xlabel("Frequency [Hs]")
		ylabel_title = r'Summed |FFT(filter)|$^2$'
		axs_a.set_ylabel(ylabel_title)
		axs_b.set_ylabel(ylabel_title)
		axs_a.set_title("Frequency FTT in the range {} to {} Hz \n Frequency coverage in the range {} to {}HZ is {} \n Redundancy (area above y=1) in the range {} to {}Hz is:  {}".format(
			round(np.min(frequency_range),1),
			round(np.max(frequency_range) ,1),
			round(np.min(freqs),1),
			round(np.max(freqs),1),
			round( (np.sum(intensities_max[interest_freq_range_indexes])/interest_freq_range_indexes.shape[0]), 5),
			round(np.min(freqs),1),
			round(np.max(freqs),1),
			round((np.sum(intensities_min[interest_freq_range_indexes]) / interest_freq_range_indexes.shape[0]) - 1, 5)
		))

		plt.close(fig)
		return(fig)
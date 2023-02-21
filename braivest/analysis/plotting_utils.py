import plotly.express as px
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
from braivest.preprocess.wavelet_utils import butter_highpass_filter

def plot_encodings(encodings, color=None, color_map=None, x_range=None, y_range=None, scatter_kwargs={}):
	"""
	Plot 2-D encodings using plotly
	Inputs:
	- 2-D encodings (dtype: array-like): the encodings to plot
	- color (dtype: list): List of the same length as encodings, each value corresponds to a category for coloring
	- color_map (dtype: dictionary): How to map the values from color to plotly-defined colors
	- x_range (dtype: tuple): X range of the plot
	- y_range (dtype: tuple): Y range of the plot
	Returns:
	- encodings figure
	"""
	if color is not None:
		if color_map is not None:
			fig = px.scatter(encodings, x=0, y=1, color = color, color_discrete_map=color_map, *scatter_kwargs)
		else:
			fig = px.scatter(encodings, x=0,y=1, color=color, *scatter_kwargs)
	else:
		fig = px.scatter(encodings, x=0, y=1, *scatter_kwargs)
	
	fig.update_traces(marker=dict(size=2, opacity=0.7))
	fig.update_layout(showlegend=False)
	if x_range:
		fig.update_xaxes(range =x_range)
	if y_range:
		fig.update_yaxes(range=y_range)
	return fig

def plot_raw_data(raw_eeg, raw_emg, raw_index, sample_rate, segment=10, highpass=None):
	"""
	Plots a section of the raw eeg and emg
	Inputs:
	- raw_eeg (dtype: np.ndarray): Raw eeg signal
	- raw_emg (dtype: np.ndarray): Raw emg signal
	- raw_index (dtype: int): Index of raw signal to plot
	- sample_rate (dtype: float): Sample rate of the raw data
	- segment (dtype: foat): How many seconds of data to plot
	- highpass(dtype:float, default: None): Value to highpass filter the data to remove slow artifacts. If none, then don't filter the data.
	Returns:
	- plot with 2 subplots: EEG on top and EMG on bottom
	"""
	plt.figure(figsize=(25, 4))
	plt.subplot(2, 1, 1)
	data = raw_eeg[raw_index - segment*sample_rate: raw_index + segment*sample_rate]
	if highpass:
		data = butter_highpass_filter(data, highpass, sample_rate)
	plt.plot(np.arange(-segment,segment,1/sample_rate), data, rasterized=True)
	plt.ylim((-0.8, 0.8))
	plt.subplot(2, 1, 2)
	data = raw_emg[raw_index - segment*sample_rate: raw_index + segment*sample_rate]
	plt.plot(np.arange(-segment,segment,1/sample_rate), data, rasterized=True)
	plt.ylim((-1, 1))
	plt.xlabel("Time(s)")
	return plt.gcf()

def get_feature_color(Pxx, f, start, stop):
	"""
	Calculates the z-scored normalized power in a certain frequency band for all points in a series. To be used as "color" parameter in plot_encodings
	Input:
	- Pxx (dtype: np.ndarray): of shape (n_samples, n_freqs), calculated power spectral densities
	- f (dtype: np.ndarray): of shape (n_freqs,): Corresponding frequencies for Pxx
	- start (dtype: int): Start index for desired frequency band in f
	- stop (dtype: int): Stop index for desired frequency band in f
	Returns:
	- np.ndarray of size (n_samples, ) of z-scored (power of band/total power)
	"""
	return zscore(np.trapz(Pxx[:, start:stop], f[start:stop], axis=1)/np.trapz(Pxx, f, axis=1))


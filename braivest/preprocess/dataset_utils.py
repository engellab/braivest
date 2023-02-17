import numpy as np

def bin_data(data, original_sample, sampling_rate):
	"""
	A helper function that reshapes the data into bins based on the original sampling rate 
	and the desired sampling rate.
	Input: 
		data: dtype: ndarray, of shape (nsamples, ) or (nsamples, wavelets)
		original_sample: dtype: float, original sampling rate
		sampling_rate: dtype: float, desired sampling rate
	Returns:
		Binned data, ndarray shape (nsamples, subfactor) or (nsamples, subfactor, nwavelets)
	"""
	# data should have shape (nsamples, ) or (nsamples, nwavelets)
	# returns (nsamples, subfactor) or (nsamples, subfactor, nwavelets)
	subfactor = original_sample/sampling_rate
	data_cut = data[:int(int(data.shape[0]/subfactor)*subfactor)]
	if subfactor > 1:
		if data_cut.ndim == 2:
			rec_sub = np.reshape(data_cut, (-1, int(subfactor), data_cut.shape[1]))
		else:
			rec_sub = np.reshape(data_cut, (-1, int(subfactor)))
		return rec_sub
	return data

def find_artifacts(signal, threshold, sample_rateio):
	"""
	A helper function to find indices of artifacts in the data based on an amplitude threshold.
	Input: 
		signal: dtype: ndarray, the signal to find the artifacts in.
		threshold: dtype: float, the threshold for artifacts
		sample_ratio: the ratio of the sample rate of the original data to the dataset indices

	"""
	indices = np.argwhere(np.abs(signal)> threshold)
	sub_indices = (indices/sample_ratio).astype(int)
	return np.unique(sub_indices)
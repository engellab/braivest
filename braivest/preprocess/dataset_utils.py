import numpy as np

def bin_data(data, original_sample, sampling_rate):
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

def find_artifacts(signal, threshold, sample_rate):
	indices = np.argwhere(np.abs(signal)> threshold)
	sub_indices = (indices/sample_rate).astype(int)
	return np.unique(sub_indices)
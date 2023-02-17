"""General utils"""
import os
import numpy as np

def load_data(cache_folder, name, allow_pickle=False):
	"""
	Loading data from a folder, essentially a wrapper around np.load
	Inputs:
	- cache_folder (dtype: str): name of folder
	- name (dtype: str): name of data file
	- allow_pickle (dtype: bool)
	Returns: np.ndarray of data loaded
	"""
	data_path = os.path.join(cache_folder, name)
	return np.load(data_path, allow_pickle=allow_pickle)
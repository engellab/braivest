import os
import numpy as np

def load_data(cache_folder, name, allow_pickle=False):
	data_path = os.path.join(cache_folder, name)
	return np.load(data_path, allow_pickle=allow_pickle)
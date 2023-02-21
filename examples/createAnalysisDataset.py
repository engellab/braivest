import numpy as np
from sklearn.preprocessing import StandardScaler
import wandb
import pickle
import os
from braivest.preprocess.wavelet_utils import get_wavelet_freqs
from braivest.preprocess.dataset_utils import find_artifacts, bin_data

sessions = [0,2,4,5,6,7,8,9,10,11,12]
train_artifact = "training_set:v0"

def load_and_log():
	names = ['sess_datas', 'sess_hypnos']
	datasets = (sess_datas, sess_hypnos)
	raw_data = wandb.Artifact(
		"analysis_set", type="dataset",
		description="probe validation sets for subject0",
		metadata={"source": "chauvette_timofeev",
					"train_set": train_artifact,
					"window_size": 2,
					"sessions": sessions,
					"wavelet_id": 6,
						"probe": "all"})
	for name, data in zip(names, datasets):
			# 🐣 Store a new file in the artifact, and write something into its contents.
		with raw_data.new_file(name + ".npy", mode="wb") as file:
			np.save(file, data)
	run.log_artifact(raw_data)

with wandb.init(project="braivest_tutorial", job_type="download") as run:
	artifact = run.use_artifact(train_artifact, type='dataset')
	artifact_dir = artifact.download()

	file = open(os.path.join(artifact_dir, "ss.pkl"), "rb")
	train_ss = pickle.load(file)

	wavelet_artifact = run.use_artifact("wavelet_data:v0")
	wavelet_artifact_dir = wavelet_artifact.download()

	sess_datas = []
	sess_hypnos = []
	for sess in sessions:
		lfp = np.load(os.path.join(wavelet_artifact_dir, "lfp_wave_session{}.npy".format(sess)))
		emg = np.load(os.path.join(wavelet_artifact_dir, "emg_wave_session{}.npy".format(sess)))
		hypno = None
		if sess == 0:
			hypno = np.load(os.path.join(wavelet_artifact_dir, "hypno.npy"))
		
		avg_emg = np.expand_dims(np.trapz(emg, get_wavelet_freqs(1, 50, 30), axis=1), -1)
		min_len = min(len(lfp), len(avg_emg))
		data = np.concatenate((lfp[:min_len], avg_emg[:min_len]), axis=1)

		ss = StandardScaler()
		data = ss.fit_transform(data)
		data[np.isnan(data)] = np.nanmax(data)

		sess_datas.append(train_ss.transform(np.mean(bin_data(data, 100, 0.5), axis=1)))
		if hypno is not None:
			sess_hypnos.append(hypno[::2])
		else:
			sess_hypnos.append([])
	load_and_log()

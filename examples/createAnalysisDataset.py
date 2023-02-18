import numpy as np
from sklearn.preprocessing import StandardScaler
import wandb
import pickle
import os
from braivest.preprocess.wavelet_utils import get_wavelet_freqs
from braivest.preprocess.dataset_utils import find_artifacts, bin_data

subject = 4
sessions = [2,3,4,5,7,9,10,11,13,14,15,16,17]
train_artifact = "allprobes_subject4_test2:v1"

with wandb.init(project="lfp_VAE", job_type="download") as run:
	artifact = run.use_artifact(train_artifact, type='dataset')
	artifact_dir = artifact.download()

file = open(os.path.join(artifact_dir, "ss.pkl"), "rb")
train_ss = pickle.load(file)

sess_datas = []
sess_hypnos = []
for sess in sessions:
	#lfp = h5 load session lfp
	#emg = h5 load session emg
	#hypno = load hypno
	
	avg_emg = np.expand_dims(np.trapz(emg, get_wavelet_freqs(1, 50, 30), axis=1), -1)
	min_len = min(len(lfp), len(avg_emg))
	data = np.concatenate((lfp[:min_len], avg_emg[:min_len]), axis=1)

	ss = StandardScaler()
	data = ss.fit_transform(data)
	data[np.isnan(data)] = np.nanmax(data)

	sess_datas.append(train_ss.transform(np.mean(bin_data(data, 100, 0.5), axis=1)))
	if hypno:
		sess_hypnos.append(hypno[::2])
	else:
		sess_hypnos.append([])

def load_and_log():
	with wandb.init(project="lfp_VAE", job_type="load-and-split-data") as run:
		names = ['sess_datas', 'sess_hypnos']
		datasets = (sess_datas, sess_hypnos)
		raw_data = wandb.Artifact(
			"subject{0}_val".format(subject), type="dataset",
			description="probe validation sets for subject0",
			metadata={"source": "chauvette_timofeev",
					  "train_set": train_artifact,
					  "window_size": 2,
						"subject": subject,
						"sessions": sessions,
						"wavelet_id": 6,
						 "probe": "all"})
		for name, data in zip(names, datasets):
				# 🐣 Store a new file in the artifact, and write something into its contents.
			with raw_data.new_file(name + ".npy", mode="wb") as file:
				np.save(file, data)
		run.log_artifact(raw_data)

load_and_log()

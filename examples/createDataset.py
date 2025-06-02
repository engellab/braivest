import numpy as np
from sklearn.preprocessing import StandardScaler
import wandb
import pickle
from braivest.preprocess.wavelet_utils import get_wavelet_freqs
from braivest.preprocess.dataset_utils import find_artifacts, bin_data
import os

def get_dataset(train_sess, test_sess, run):
	train = np.empty((0, 31))
	train_Y = np.empty((0, 31))
	test = np.empty((0, 31))
	test_Y = np.empty((0, 31))
	hypnos = []
	for sess in np.append(train_sess, test_sess):
			wavelet_artifact = run.use_artifact("wavelet_data:v0")
			wavelet_artifact_dir = wavelet_artifact.download()
			lfp_wavelet = np.load(os.path.join(wavelet_artifact_dir, "lfp_wave_session{}.npy".format(sess)))
			emg_wavelet = np.load(os.path.join(wavelet_artifact_dir, "emg_wave_session{}.npy".format(sess)))
			hypno = None
			if sess == 0:
				hypno = np.load(os.path.join(wavelet_artifact_dir, "hypno.npy"))

			raw_artifact = run.use_artifact("raw_data:v0")
			raw_artifact_dir = raw_artifact.download()
			raw_lfp = np.load(os.path.join(raw_artifact_dir, "lfp_session{}.npy".format(sess)))

			avg_emg = np.expand_dims(np.trapz(emg_wavelet, get_wavelet_freqs(1, 50, 30), axis=1), -1)
			min_len = min(len(lfp_wavelet), len(avg_emg))
			data = np.concatenate((lfp_wavelet[:min_len], avg_emg[:min_len]), axis=1)
			artifacts = find_artifacts(raw_lfp)
			ss = StandardScaler()
			data = ss.fit_transform(data)
			data[np.isnan(data)] = np.nanmax(data)
			data_bin = np.mean(bin_data(data, 100, 0.5),axis=1)
			artifacts_X = np.unique(np.append(artifacts, artifacts-1))
			artifacts_X = artifacts_X[artifacts_X < data_bin.shape[0]-2]
			if sess in test_sess:
				if hypno:
					hypno = hypno[::2]
					hypno = np.delete(hypno[:-1], artifacts_X)
				else:
					hypno = None
				hypnos.append(hypno)
				test = np.append(test, np.delete(data_bin[:-1], artifacts_X, axis=0), axis=0)
				test_Y = np.append(test_Y, np.delete(data_bin[1:], artifacts_X, axis=0), axis=0)
			else:
				train = np.append(train, np.delete(data_bin[:-1], artifacts_X, axis=0), axis=0)
				train_Y = np.append(train_Y, np.delete(data_bin[1:], artifacts_X, axis=0), axis=0)
	ss = StandardScaler()
	train = ss.fit_transform(train)
	test = ss.transform(test)
	train_Y = ss.transform(train_Y)
	test_Y = ss.transform(test_Y)

	return train, train_Y, test, test_Y, hypnos, ss

# This is for uploading to wandb as an artifact.
# If you don't want to use wandb, you can remove this function and use the get_dataset function directly.
def load_and_log(train_sess, test_sess, subject, probes):
	with wandb.init(project="braivest_tutorial", job_type="load-and-split-data") as run:
		names = ['train', 'train_Y', 'test','test_Y']
		train, train_Y, test, test_Y, hypno, ss = get_dataset(train_sess, test_sess, run)
		datasets = (train, train_Y, test, test_Y)
		raw_data = wandb.Artifact(
			"training_set".format(subject, test_sess[0]), type="dataset",
			description="LFP/MEG from subject {0}".format(subject),
			metadata={"source": "chauvette_timofeev",
					  "window_size": 2,
						"subject": 0,
						"train_sessions": train_sess,
						"test_sessions": test_sess,
						 "probe": 'visual',
						 "wave_id": 6})
		for name, data in zip(names, datasets):
				# 🐣 Store a new file in the artifact, and write something into its contents.
			with raw_data.new_file(name + ".npy", mode="wb") as file:
				np.save(file, data)
		if hypno is not None:
			with raw_data.new_file("hypno.npy", mode = "wb") as file:
				np.save(file, hypno)
		with raw_data.new_file("ss.pkl", mode="wb") as file:
			pickle.dump(ss, file)
		run.log_artifact(raw_data)


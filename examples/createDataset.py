import numpy as np
from sklearn.preprocessing import StandardScaler
import wandb
import pickle
from braivest.preprocess.wavelet_utils import get_wavelet_freqs
from braivest.preprocess.dataset_utils import find_artifacts, bin_data

def get_dataset(train_sess, test_sess):
	train = np.empty((0, 31))
	train_Y = np.empty((0, 31))
	test = np.empty((0, 31))
	test_Y = np.empty((0, 31))
	hypnos = []
	for sess in np.append(train_sess, test_sess):
			#lfp = h5 load session lfp
			#emg = h5 load session emg
			#hypno = load hypno
			avg_emg = np.expand_dims(np.trapz(emg, get_wavelet_freqs(1, 50, 30), axis=1), -1)
			min_len = min(len(lfp), len(avg_emg))
			data = np.concatenate((lfp[:min_len], avg_emg[:min_len]), axis=1)
			artifacts = find_artifacts(lfp)
			ss = StandardScaler()
			data = ss.fit_transform(data)
			data[np.isnan(data)] = np.nanmax(data)
			data_bin = np.mean(bin_data(data, 100, 0.5),axis=1)
			artifacts_X = np.unique(np.append(artifacts, artifacts-1))
			artifacts_X = artifacts_X[artifacts_X < data_bin.shape[0]-2]
			if sess in test_sess:
				if curr_hypno:
					hypno = curr_hypno[::2]
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

def load_and_log(train_sess, test_sess, subject, probes):
	with wandb.init(project="lfp_VAE", job_type="load-and-split-data") as run:
		names = ['train', 'train_Y', 'test','test_Y']
		train, train_Y, test, test_Y, hypno, ss = get_dataset(subject, probes, train_sess, test_sess)
		datasets = (train, train_Y, test, test_Y)
		raw_data = wandb.Artifact(
			"probe6_subject{0}_test{1}".format(subject, test_sess[0]), type="dataset",
			description="LFP/MEG from subject {0}, all probes".format(subject),
			metadata={"source": "chauvette_timofeev",
					  "window_size": 2,
						"subject": subject,
						"train_sessions": train_sess,
						"test_sessions": test_sess,
						 "probe": 'all',
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


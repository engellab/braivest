import tensorflow as tf
import numpy as np
import sys
import wandb
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
from wandb_callbacks import *
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from braivest.model.emgVAE import emgVAE
from braivest.utils import load_data


class Trainer():
	def __init__(self, config, input_dim, test_size=0.2, ):
		self.config = config
		layers = [config.layer_dims for layer in range(config.num_layers)]
		self.model = emgVAE(input_dim, config.latent, layers, config.kl, config.nw, emg = config.emg)
		self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), loss='mse', metrics = ['mse'])

	def load_artifact(self, artifact_dir):
		if self.config.time:
			train_X = load_data(artifact_dir, 'train.npy')
			train_Y = load_data(artifact_dir, 'train_Y.npy')
			train_set = (train_X, train_Y)
		else:
			train_X = load_data(artifact_dir, 'train.npy')
			train_set = (train_X, train_X)
		x_train, x_val, y_train, y_val = train_test_split(train_set[0], train_set[1], test_size=test_size, shuffle=True)
		self.train_set = (x_train, y_train)
		self.val_set = (x_val, y_val)
		try:
			self.hypno = load_data(artifact_dir, 'hypno.npy')
		except Error e:
			self.hypno = None
	
	def train(self, train_set=None, val_set=None, wandb=False, custom_callback=False):
		assert train_set is not None or self.train_set is not None, "No training data supplied."
		if train_set is not None:
			self.train_set = train_set
		if val_set is not None:
			self.val_set = val_set
		if wandb:
			wandb_callback = WandbCallback(save_model=False)
			if custom_callback:
				custom_callback = CustomWandbCallback(val_set, val_hypno, plot=False)
				callbacks = [wandb_callback, custom_callback]
			else:
				callbacks = [wandb_callback]
		else:
			callbacks = None
		history = self.model.fit(train_set[0], train_set[1], epochs=self.config.num_epochs, batch_size=self.config.batch_size,
				validation_data=val_set, callbacks=callbacks)
		return history.history

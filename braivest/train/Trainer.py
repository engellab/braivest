import tensorflow as tf
import numpy as np
import sys
import wandb
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
from braivest.train.wandb_callbacks import *
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from braivest.model.emgVAE import emgVAE
from braivest.utils import load_data


class Trainer():
	"""
	Trainer class to train the braivest model.
	"""
	def __init__(self, config, input_dim):
		"""
		Input:
			config (dtype: dictionary): the config file for training the model, provided by wandb or as a json
				See example in examples/config.json
			input_dim (dtype: int): The dimension of the input data
			test_size (dtype: float): The fraction of data to 
		"""
		self.config = config
		layers = [config.layer_dims for layer in range(config.num_layers)]
		self.model = emgVAE(input_dim, config.latent, layers, config.kl, emg = config.emg)
		self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), loss='mse', metrics = ['mse'])

	def load_dataset(self, dataset_dir, val_size=0.2):
		"""
		Load a dataset from a dataset directory
		Input: 
			dataset_dir (dtype: string): The relative path of the dataset directory
			val_size (dtype: float, default=0.2): The percent of the data to use as a validation set
		"""
		if self.config.time:
			train_X = load_data(dataset_dir, 'train.npy')
			train_Y = load_data(dataset_dir, 'train_Y.npy')
			train_set = (train_X, train_Y)
		else:
			train_X = load_data(dataset_dir, 'train.npy')
			train_set = (train_X, train_X)
		x_train, x_val, y_train, y_val = train_test_split(train_set[0], train_set[1], test_size=val_size, shuffle=True)
		self.train_set = (x_train, y_train)
		self.val_set = (x_val, y_val)
		try:
			self.hypno = load_data(artifact_dir, 'hypno.npy')
		except Exception:
			self.hypno = None
	
	def train(self, train_set=None, val_set=None, wandb=False, custom_callback=False, save_model=True):
		"""
			Train the model.
			Input:
				train_set (dtype: tuple of (train_X, train_Y)): If a dataset was not loaded with the artifact, a training set can be provided.
				val_set (dtype: tuple of (val_X, val_Y)): optional validation set. 
				wandb (dtype: boolean): Whether to use wandb to train the model
				custom_callback (dtype: boolean): Whether to use the custom callback in wandb_callbacks
				save_model (dtype: boolean): Whether or not to save the model every epoch.

		"""
		assert train_set is not None or self.train_set is not None, "No training data supplied."
		if train_set is not None:
			self.train_set = train_set
		if val_set is not None: s
			self.val_set = val_set
		if wandb:
			wandb_callback = WandbCallback(save_model=save_model)
			if custom_callback:
				custom_callback = CustomWandbCallback(val_set, self.hypno, plot=False)
				callbacks = [wandb_callback, custom_callback]
			else:
				callbacks = [wandb_callback]
		else:
			callbacks = None
		history = self.model.fit(self.train_set[0], self.train_set[1], epochs=self.config.epochs, batch_size=self.config.batch_size,
				validation_data=self.val_set, callbacks=callbacks)
		return history.history

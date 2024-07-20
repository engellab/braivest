import tensorflow as tf
import tf_keras as keras
import numpy as np
import sys
import wandb
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
import os
from braivest.model.emgVAE import emgVAE
from braivest.utils import load_data


"""
kwargs...

if i have a function f(a, b, **kwargs), then inside f, kwargs is a dictionary..
so i can call it like this f(1, 2, a=2, b=2) and then kwargs inside is {"a":2, ...}

*args ....

give f(*args), then you can call it like this f(1,2,3,4) and then args is a list.
"""


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
		layers = [config['layer_dims'] for layer in range(config['num_layers'])]
		self.model = emgVAE(input_dim, config['latent'], layers, config['kl'], emg = config['emg'])
		self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=config['lr']), loss='mse', metrics = ['mse'])

	def load_dataset(self, dataset_dir, val_size=0.2):
		"""
		Load a dataset from a dataset directory
		Input: 
			dataset_dir (dtype: string): The relative path of the dataset directory. Should contain file "train.npy", and if applicable, "train_Y.npy" and "hypno.npy"
			val_size (dtype: float, default=0.2): The percent of the data to use as a validation set
		"""
		if self.config['time']:
			train_X = load_data(dataset_dir, 'train.npy')
			train_Y = load_data(dataset_dir, 'train_Y.npy')
			train_set = (train_X, train_Y)
		else:
			train_X = load_data(dataset_dir, 'train.npy')
			train_set = (train_X, train_X)
		x_train, x_val, y_train, y_val = train_test_split(train_set[0], train_set[1], test_size=val_size, shuffle=True)
		self.train_set = (x_train, y_train)
		self.val_set = (x_val, y_val)
	
	def train(self, train_set=None, val_set=None, wandb=False, save_model=True, save_best_only=True, save_dir = None, train_kwargs = {}, save_kwargs = {}, custom_callbacks=[]):
		"""
			Train the model.
			Input:
				train_set (dtype: tuple of (train_X, train_Y)): If a dataset was not loaded with the artifact, a training set can be provided.
				val_set (dtype: tuple of (val_X, val_Y)): optional validation set. 
				wandb (dtype: boolean): Whether to use wandb to train the model
				save_model (dtype: boolean): Whether or not to save the model every epoch.
				train_kwargs (dtype: dict)
				save_kwargs (dtype: dict)
		NOTE: The way wandb is saving model is not compatible with tf-probability. To save model, supply wandb.run.dir to save_dir
		"""
		assert train_set is not None or self.train_set is not None, "No training data supplied."
		if train_set is not None:
			self.train_set = train_set
		if val_set is not None: 
			self.val_set = val_set
		callbacks = custom_callbacks
		if wandb:
			wandb_callback = WandbCallback(save_model=False)
			callbacks.append(wandb_callback)
		if save_model:
			if save_dir is None:
					print("No save directory provided!")
			else:
				save_callback = keras.callbacks.ModelCheckpoint(os.path.join(save_dir, "model_{epoch:02d}.h5"), save_weights_only=True, save_best_only=save_best_only, **save_kwargs)
				callbacks.append(save_callback)
		history = self.model.fit(self.train_set[0], self.train_set[1], epochs=self.config['epochs'], batch_size=self.config['batch_size'],
				validation_data=self.val_set, callbacks=callbacks, **train_kwargs)
		return history.history

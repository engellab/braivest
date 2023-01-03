import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np

class emgVAE(keras.Model):
	def __init__(self, input_dim, latent_dim, hidden_states, kl, emg = True, sample_weights = False):
		super(emgVAE, self).__init__()
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim))
		self.encoder = tf.keras.Sequential()
		self.encoder.add(tf.keras.layers.InputLayer(input_shape=(self.input_dim,)))
		for n_units in hidden_states:
			self.encoder.add(tf.keras.layers.Dense(n_units, activation='relu'))

		self.encoder.add(tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.latent_dim),activation=None, name='z'))
		self.encoder.add(tfp.layers.MultivariateNormalTriL(self.latent_dim , convert_to_tensor_fn=tfp.distributions.Distribution.sample, name='z_layer'))
		self.encoder.add(tfp.layers.KLDivergenceAddLoss(self.prior, weight=kl))
		self.decoder = tf.keras.Sequential()
		for n_units in hidden_states:
			self.decoder.add(tf.keras.layers.Dense(n_units, activation='relu'))
		self.decoder.add(tf.keras.layers.Dense(self.input_dim))


		self.loss_tracker = keras.metrics.Mean(name="loss")
		self.neighbor_loss_tracker = keras.metrics.Mean(name="neighbor_loss")
		self.mse = keras.metrics.MeanSquaredError(name='mse')
		self.emg = emg

	def call(self, inputs):
		z = self.encode(inputs)
		return self.decoder(z)

	def encode(self, inputs):
		inputs = tf.cast(inputs, dtype=tf.float32)
		z = self.encoder(inputs)
		if self.emg:
			temp = tf.concat((z, tf.expand_dims(inputs[:, -1], 1)), axis=1)
			return temp
		return z


	def train_step(self, data):
		# Unpack the data. Its structure depends on your model and
		x, y = data

		with tf.GradientTape() as tape:
			# Compute the loss value
			# (the loss function is configured in `compile()`)
			y_pred = self(x, training=True)
			in_encodings = self.encode(x)
			out_encodings = self.encode(y)
			neighbor_loss = tf.math.reduce_mean(tf.math.reduce_euclidean_norm(in_encodings - out_encodings, axis=1))
			loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

		# Compute gradients
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)
		# Update weights
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))
		# Update metrics (includes the metric that tracks the loss)
		self.loss_tracker.update_state(loss)
		self.neighbor_loss_tracker.update_state(neighbor_loss)
		self.mse.update_state(y, y_pred)
		# Return a dict mapping metric names to current value
		metrics = {m.name: m.result() for m in self.metrics}
		metrics['loss'] = self.loss_tracker.result()
		metrics['neighbor_loss'] = self.neighbor_loss_tracker.result()
		return metrics

	@property
	def metrics(self):
		return [self.loss_tracker, self.neighbor_loss_tracker, self.mse]

	def test_step(self, data):
		x, y = data
		y_pred = self(x, training=False)
		in_encodings = self.encode(x)
		out_encodings = self.encode(y)
		neighbor_loss = tf.math.reduce_mean(tf.math.reduce_euclidean_norm(in_encodings - out_encodings, axis=1))/tf.math.reduce_mean(tf.math.reduce_euclidean_norm(in_encodings, axis=1))
		loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
		self.loss_tracker.update_state(loss)
		self.neighbor_loss_tracker.update_state(neighbor_loss)
		self.mse.update_state(y, y_pred)
		metrics = {m.name: m.result() for m in self.metrics}
		metrics['loss'] = self.loss_tracker.result()
		metrics['neighbor_loss'] = self.neighbor_loss_tracker.result()
		return metrics

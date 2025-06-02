import wandb
import plotly.express as px
import tensorflow as tf
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import metrics
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.stats import skew, kurtosis
import tf_keras as keras
from keras.losses import MeanSquaredError
from scipy.stats import entropy

def distribution_metrics_labels(encodings, hypno):
    length = min(len(hypno), encodings.shape[0])
    encodings = encodings[:length, :]
    ks = []
    sks = []
    hypno = np.asarray(hypno)[:length]
    hypno_unique = np.unique(hypno)
    for label in hypno_unique:
        label_encodings = encodings[hypno == label, :]
        ks.append(kurtosis(label_encodings, axis=0))
        sks.append(np.abs(skew(label_encodings, axis=0)))

    labels_true = hypno
    silhouette = metrics.silhouette_score(encodings, labels_true)
    return {'skew': np.mean(sks), 'kurtosis': np.mean(ks), 'silhouette': silhouette}


def get_entropy(encodings):
     encodings_1d = np.expand_dims(encodings[:, 0], -1)
     kde = KernelDensity().fit(encodings_1d)
     log_density = kde.score_samples(encodings_1d)
     return entropy(np.exp(log_density))


def plot_encodings(encodings, title, hypno=None):
        hypno_unique = np.unique(hypno)
        #legend = {hypno_unique[0]:'REM',hypno_unique[1]:'SWS',hypno_unique[2]:'Wake'}
        #if hypno_unique.shape[0] > 3:
            #legend[hypno_unique[3]] = 'X'
        if hypno is not None:
            length = min(len(hypno), encodings.shape[0])
            if encodings.shape[1] == 3:
                fig = px.scatter_3d(encodings[:length, :], x=0, y=1, z=2, color=hypno[:length])
            else:
                fig = px.scatter(encodings[:length, :], x=0, y=1, color=hypno[:length])
            #fig = px.scatter(encodings, x=0, y=1, color = [legend[i] for i in hypno], labels = legend)
        else:
            if encodings.shape[1] == 3:
                fig  = px.scatter_3d(encodings, x=0, y=1, z=2)
            else:
                fig = px.scatter(encodings, x=0, y=1)
        fig.update_traces(marker=dict(size=5, opacity=0.5))
        return fig

class CustomWandbCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, hypno, plot = False):
        self.hypno = hypno
        self.validation_data = validation_data
        self.plot = plot

    def on_epoch_end(self, epoch, logs=None):
        if epoch < 10 or epoch % 20 == 0:
            encodings = self.get_encodings()
            if self.plot:
                fig = plot_encodings(encodings, "Val Plot", hypno=self.hypno)
                wandb.log({"epoch": epoch, "encodings": fig}, commit=False)
            if self.hypno is not None:
                wandb.log(distribution_metrics_labels(encodings, self.hypno), commit=False)
            wandb.log({'entropy': get_entropy(encodings)}, commit=False)

    def get_encodings(self):
        print(self.validation_data[0].shape)
        encodings = self.model.encode(self.validation_data[0])
        encodings = tf.convert_to_tensor(encodings).numpy()
        return encodings

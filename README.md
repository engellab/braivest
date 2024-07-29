# braivest - BRAIn STate VAE

Welcome to braivest! Braivest is a package for analyzing brain state data based on Variational Autoencoders (VAEs). 
Accompanying preprint is coming soon!

Why use braivest?

1) Supervised learning for brain state classification is restricted to predefined states and biased by expert labels
2) Traditional dimensionality reduction/visualization methods fail
3) Braivest can be used to compare brain state data across... experimental conditions (i.e. healthy vs cancerous), brain regions, task conditions, and more!
4) Braivest is agnostic to recording modality including LFP, EEG, voltage imaging, and more!

## Setup & Installation

### Environment

We recommend using a conda or virtualenv environment for braivest. Environment yaml file is provided. In addition, please install tensorflow and tensorflow-probability with instructions from the website. Currently, only tensorflow 2.15.0 and tensorflow-probability 0.23.0 are supported. Higher versions have keras dependency issues.

### GPU setup

We recommend using a machine with access to GPU nodes for training of models. Our models are written using Tensorflow and Tensorflow probability. Please refer to the Tensorflow documentation to learn how to set up your GPU for use.

### Installing the package

```
pip install .
```

## Preprocessing your data

The first step to using braivest is to preprocess your data.
First use [pywavelets](https://pywavelets.readthedocs.io/en/latest/ref/index.html) to calculate wavelets from your data. Be sure to take note of the sampling rate of your data. Use [wavelet_utils](braivest.preprocess.wavelet_utils) to determine scales and calculate continuous wavelets, then transforming them into power spectral density data. NOTE: Wavelets are computationally expensive and calculations will be much faster with multithreading.

Then, turn your transformed data into a usable dataset. Take consideration of training vs test set. You can look at examples in [examples](examples).
The emgVAE (optionally) uses the last axis of input data (traditionally emg), as one of the axes of the low-dimensional manifold. 

## Wandb Integration

We recommend using [wandb](https://docs.wandb.ai/quickstart) for dataset versioning, hyperparameter sweeps, and model training tracking. Our package includes wandb flags for optionally using wandb.

## Model Training
After you have your dataset, it is time to train your model using the Trainer. Follow the [example](examples/train_wandb.py)

## Training an HMM
After obtaining your low-dimensional manifold representation, you can optionally segment your data using a Hidden Markov Model (HMM). We recommend using the [ssm package](github.com/linderman/ssm) from Linderman Lab . 
Additionally, we have forked the linderman/ssm repository to add an additional type of HMM which allows for multiple transition graphs for different datasets that share cluster emissions (termed MultiHMM). You can find that [here](github.com/engellab/ssm)

## Analysis
There are several types of analysis that can be done. Follow examples in the [examples section](examples).

## Interactive Plots
Plotly allows for interactive web apps to view data. We have included [example apps](interactive) for viewing the manifold, selecting a point, and viewing the corresponding raw data or spectrogram. 











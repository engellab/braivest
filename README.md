# braivest - BRAIn STate VAE

Welcome to braivest! Braivest is a package for analyzing brain state data based on Variational Autoencoders (VAEs). Learn more by reading our paper here:

Why use braivest?

1) Supervised learning for brain state classification is restricted to predefined states and biased by expert labels
2) Traditional dimensionality reduction/visualization methods fail
3) Braivest can be used to compare brain state data across... experimental conditions (i.e. healthy vs cancerous), brain regions, task conditions, and more!
4) Braivest is agnostic to recording modality including LFP, EEG, voltage imaging, and more!

## Setup & Installation

Environment

GPU setup

Data storage


## Preprocessing your data

The first step to using braivest is to preprocess your data.
First use pywavelets to calculate wavelets from your data. Be sure to take note of the sampling rate of your data. Use braivest.preprocess.wavelet_utils to determine scales and calculate continuous wavelets, then transforming them into power spectral density data.

Then, turn your transformed data into a usable dataset. Take consideration of training vs test set. 
The emgVAE uses the last axis of input data (traditionally emg), as one of the axes of the low-dimensional manifold. 

## Train the model

Once you have your dataset, train the model. 

## Analysis








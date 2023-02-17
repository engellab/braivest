import plotly.express as px
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
from braivest.preprocess.wavelet_utils import butter_highpass_filter

def plot_encodings(encodings, hypno=None, color=None, color_map=None, x_range=None, y_range=None):
    if hypno:
        hypno_unique = np.unique(hypno)
        legend = {hypno_unique[0]:'REM',hypno_unique[1]:'SWS',hypno_unique[2]:'Wake'}
        color_map = {'REM':"#0000ff", "Wake":"#ff0000", "SWS":"#00ff00"}
        color = [legend[i] for i in hypno]
    if color:
        if color_map:
            fig = px.scatter(encodings, x=0, y=1, color = color, color_discrete_map=color_map)
        else:
            fig = px.scatter(encodings, x=0,y=1, color=color)
    else:
        fig = px.scatter(encodings, x=0, y=1)
    
    fig.update_traces(marker=dict(size=2, opacity=0.7))
    fig.update_layout(showlegend=False)
    if x_range:
        fig.update_xaxes(range =x_range)
    if y_range:
        fig.update_yaxes(range=y_range)
    return fig

def plot_raw_data(raw_eeg, raw_emg, raw_index, sample_rate, segment=10, highpass=None):
    plt.figure(figsize=(25, 4))
    plt.subplot(2, 1, 1)
    data = raw_eeg[raw_index - segment*sample_rate: raw_index + segment*sample_rate]
    if highpass:
        data = butter_highpass_filter(data, highpass, sample_rate)
    plt.plot(np.arange(-segment,segment,1/sample_rate), data, rasterized=True)
    plt.ylim((-0.8, 0.8))
    plt.subplot(2, 1, 2)
    data = raw_emg[raw_index - segment*sample_rate: raw_index + segment*sample_rate]
    plt.plot(np.arange(-segment,segment,1/sample_rate), data, rasterized=True)
    plt.ylim((-1, 1))
    return plt.gcf()

def get_feature_color(Pxx, f, start, stop):
    return zscore(np.trapz(Pxx[:, start:stop], f[start:stop], axis=1)/np.trapz(Pxx[:, :13], f[:13], axis=1))


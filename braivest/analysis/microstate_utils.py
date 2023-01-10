from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import mode
import numpy as np
import seaborn as sns
from hmm_utils import get_labels
import matplotlib.pyplot as plt

def get_microstates(sess_labels):
    states = {}
    for i in range(9):
        for j in range(9):
            states[(i,j)] = []
    windows = sliding_window_view(sess_labels, 15)
    modes, counts = mode(windows, axis=1)
    for i in range(7, len(windows) - 7):
        if counts[i-7] > 10:
            if modes[i-7] != sess_labels[i]:
                states[(modes[i-7][0], sess_labels[i])].append(i)
    return states

def get_transitions(labels_sess, state1, state2):
    modes, counts = mode(labels_sess)
    modes = modes.flatten()
    counts = counts.flatten()
    windows = sliding_window_view(modes, 20)
    window_modes, window_counts = mode(windows, axis=1)
    transitions = []
    for i in range(15, len(windows)-10):
        if window_modes[i-10] == state1 and window_counts[i-10] > 10 and window_modes[i+10] == state2 and window_counts[i+10] > 10:
            transitions.append(i+10)
    return transitions

def plot_microstates_table(num_states, sess_labels, mapping=None, labels=None):
    states_table = np.zeros((num_states, num_states))
    states_labels_flat =  np.concatenate(sess_labels, axis=None)
    states_indices = get_microstates(states_labels_flat)
    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                states_table[i,j] = len(states_indices[(i, j)])/np.sum(states_labels_flat==i)
    if mapping:
        states_mapped = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(num_states):
                states_mapped[mapping[i], mapping[j]] = states_table[i, j]
    else:
        states_mapped = states_table
    sns.heatmap(states_mapped*100, annot=True, xticklabels=labels, yticklabels=labels, cmap = 'Greys')
    return states_mapped, plt.gcf()

def get_microstates_probes(subject_sess, model, hmm):
    microstates = []
    for probe in range(14):
        labels = get_labels(subject_sess[probe], model, hmm, probe)
        microstates_indices = get_microstates(labels)
        microstates.append(microstates_indices)
    return microstates

def get_comicrostates(microstates, micro_type, mapping=None, num_probes=14):
    comicrostates = np.zeros((num_probes, num_probes))
    for i in range(num_probes):
        for j in range(num_probes):
            probe1 = microstates[i][micro_type]
            probe2 = microstates[j][micro_type]
            if len(np.union1d(probe1, probe2)) != 0:
                comicrostates[mapping[i], mapping[j]] = len(np.intersect1d(probe1, probe2))/len(np.union1d(probe1, probe2))
    return comicrostates


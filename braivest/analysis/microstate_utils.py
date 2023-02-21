from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import mode
import numpy as np
import seaborn as sns
from braivest.analysis.hmm_utils import get_hmm_labels
import matplotlib.pyplot as plt

def get_microstates(sess_labels, num_states, window_size=15, ratio=2/3):
    """
    Get microstates from continuous session label
    Inputs:
     - sess_labels (dtype: list of int): list of continuous session labels
     - num_states (dtype: int): number of states
     - window_size (dtype: int, default: 15): size of window to define surrounding state
     - ratio (dtype: float, default: 2/3): minimum fraction of window in one state to be considered surrounding state
    Returns:
        - Dictionary where each key is a pair of (surrounding state, microstate) and each value is a list of indices of such microstate type
    """
    states = {}
    for i in range(num_states):
        for j in range(num_states):
            states[(i,j)] = []
    windows = sliding_window_view(sess_labels, window_size)
    modes, counts = mode(windows, axis=1)
    for i in range(int(window_size/2), int(len(windows) - window_size/2)):
        if counts[int(i-window_size/2)] > ratio*window_size:
            if modes[int(i-window_size/2)] != sess_labels[i]:
                states[(modes[int(i-window_size/2)][0], sess_labels[i])].append(i)
    return states

def get_transitions(sess_labels, state1, state2, window_size=20):
    """
    Get indices of a specific transition of state1 to state2 from continuous session label
    Inputs:
    - sess_labels (dtype: list of int): list of continuous session labels
    - state1 (dtype: int): The state label that the transition is from
    - state2 (dtype: int): The state label that the transition is to
    - window_size (dtype: int, default=20): Window size to find transitions
    Returns:
     - list of indices of transition
    """
    modes, counts = mode(sess_labels)
    modes = modes.flatten()
    counts = counts.flatten()
    windows = sliding_window_view(modes, window_size)
    window_modes, window_counts = mode(windows, axis=1)
    transitions = []
    for i in range(15, int(len(windows)-window_size/2)):
        if (window_modes[int(i-window_size/2)] == state1 and 
            window_counts[int(i-window_size/2)] > window_size/2 and 
            window_modes[int(i+window_size/2)] == state2 and 
            window_counts[int(i+window_size/2)] > window_size/2):
            transitions.append(int(i+window_size/2))
    return transitions

def plot_microstates_table(sess_labels, num_states, window_size=15, ratio=2/3, mapping=None, labels=None):
    """
    Plot a table showing the frequency of types microstates, where rows are microstates states and columns are surrounding states
    Inputs:
     - sess_labels (dtype: list of int): list of continuous session labels
     - num_states (dtype: int): number of states
     - window_size (dtype: int, default: 15): size of window to define surrounding state
     - ratio (dtype: float, default: 2/3): minimum fraction of window in one state to be considered surrounding state
     - mapping (dtype: list): mapping to reorder the states in the plot
     - labels (dtype: list): List of strings to label to table rows & columns
    Returns:
    - table of microstate frequencies
    - figure
    """
    states_table = np.zeros((num_states, num_states))
    states_indices = get_microstates(sess_labels, num_states, window_size, ratio)
    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                states_table[i,j] = len(states_indices[(i, j)])/np.sum(sess_labels==i)
    if mapping:
        states_mapped = np.zeros((num_states, num_states))
        for i in range(num_states):
            for j in range(num_states):
                states_mapped[mapping[i], mapping[j]] = states_table[i, j]
    else:
        states_mapped = states_table
    sns.heatmap(states_mapped*100, annot=True, xticklabels=labels, yticklabels=labels, cmap = 'Greys')
    return states_mapped, plt.gcf()

def get_microstates_probes(subject_sess, model, hmm, num_probes):
    """
    Get all types of microstates from all probes for a session.
    Inputs:
    - subject_sess (dtype: list): A list of session data for each probe.
    - model: The trained model
    - HMM : The trained MultiHMM
    - num_probes (dtype: int): the number of probes
    Returns:
    - list of dictionaries containing microstate indices
    """
    microstates = []
    for probe in range(num_probes):
        labels = get_hmm_labels(subject_sess[probe], model, hmm, probe)
        microstates_indices = get_microstates(labels, hmm.K)
        microstates.append(microstates_indices)
    return microstates

def get_comicrostates(microstates, micro_type, mapping=None, num_probes=14):
    """
    Get comicrostates heatmap for a certain microstate type
    Inputs:
    - microstates (dtype: list of dictionaries): Precalculated microstate indices from get_mirostates_probes
    - microtype (dtype: tuple): The microstate type (i.e. (0, 1))
    - mapping (dtype: list, default=None): mapping to determine order of rows/cols in heatmap
    - num_probes (dtype: int, default=14): number of probes
    Returns:
    - heatmap of comicrostates as np.ndarray
    """
    comicrostates = np.zeros((num_probes, num_probes))
    for i in range(num_probes):
        for j in range(num_probes):
            probe1 = microstates[i][micro_type]
            probe2 = microstates[j][micro_type]
            if len(np.union1d(probe1, probe2)) != 0:
                comicrostates[mapping[i], mapping[j]] = len(np.intersect1d(probe1, probe2))/len(np.union1d(probe1, probe2))
    return comicrostates



import numpy as np
import sys
from braivest.model.emgVAE import emgVAE
from ssm.hmm import MultiHMM, HMM
import ssm
import matplotlib.pyplot as plt
from pyvis.network import Network
import seaborn as sns

def hmm_cross_val(clusters, datasets, n_repeats=3, ind_mask=None):
"""
Cross validation for training of HMM
Inputs:
- clusters (dtype: int): The number of clusters
- datasets (dtype: list): List of datasets (continuous 2-D encodings)
- n_repeats (dtype: int, default: 3): Number of repeats to do cross-validation
- ind_mask: (dtype: array-like, default: None) Array that specifies for each dataset which probe it is from for training MultiHMM only.
Returns:
    - The trained hmm
    - List of train scores (log-likelihood)
    - List of test scores (log-likelihood)
"""
    train_scores = []
    test_scores = []
    for repeat in range(n_repeats):
        train_inds = np.random.choice(len(datasets), size=int(len(datasets)*0.8), replace=False).astype("int")
        train_data = [datasets[i] for i in train_inds]
        test_data = [datasets[i] for i in range(len(datasets)) if i not in train_inds]
        if ind_mask:
            test_mask = [ind_mask[i] for i in range(len(datasets)) if i not in train_inds]
            train_mask = [ind_mask[i] for i in train_inds]
            hmm = MultiHMM(K=clusters, D=2, N=3)
            hmm.fit(train_data, ind_mask=train_mask, method="em")
        else:
            hmm = HMM(K=clusters, D=2)
            hmm.fit(train_data, method="em", num_iters=50, init_method="kmeans")
        train_scores.append(hmm.log_likelihood(train_data, ind_mask=train_mask)/np.sum([len(train_data[i]) for i in range(len(train_data))]))
        test_scores.append(hmm.log_likelihood(test_data, ind_mask = test_mask)/np.sum([len(test_data[i]) for i in range(len(test_data))]))
    return hmm, train_scores, test_scores

def get_hmm_labels(hmm, encodings_list, trans_ind=None):
"""
Predicts hmm labels of a list of encodings
Inputs:
- hmm (dtype: HMM or MultiHMM): the HMM
- encodings_list (dtype: list of np.ndarray): list of continuous session encodings to predict labels
- trans_ind (dtype: ind, default: None): The index of transition matrix for MultiHMM
Returns:
- list of labels for each encoding session
"""
    sess_labels = []
    for split in encodings_list:
        if trans_ind:
            sess_labels.append(hmm.most_likely_states(split, trans_ind: trans_ind))
        else:
            sess_labels.append(hmm.most_likely_states(split))
    return sess_labels

def plot_state_durations(hmm, encodings_list, trans_ind= None, ordering=None, color_list=None, binwidth=0.4, kde_kws=None):
    """
    Plot the state durations
    Inputs:
    - hmm (dtype: HMM or MultiHMM): the HMM
    - encodings_list (dtype: list of np.ndarray): list of continuous session encodings to predict labels
    - trans_ind (dtype: ind, default: None): The index of transition matrix for MultiHMM
    - ordering (dtype: list): how to order the graphs for subplots
    - color_list (dtype: list of colors): color to assign each state
    - binwidth (dtype: float): bin width for kde histogram
    - kde_kwargs (dtype: dict): args to pass to seaborn kde
    Returns:
    - inferred durations 
    - state duration figure
    """
    sess_labels = get_hmm_labels(hmm, encodings_list, trans_ind)
    sess_labels = np.concatenate(sess_labels, axis=None)
    inferred_state_list, inferred_durations = ssm.util.rle(np.asarray(sess_labels))
    plt.figure(figsize=(10, 5))
    if ordering is None:
        ordering = list(range(hmm.K))
    for i, s in enumerate(ordering):
        plt.subplot(hmm.K, 1, i + 1)
        sns.histplot(np.log(inferred_durations[inferred_state_list == s]), kde=True, stat='probability', color=color_list[s], binwidth=binwidth , kde_kws=kde_kws)
        plt.xlim((0,7))
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("")
    return inferred_durations, plt.gcf()

def plot_transition_graph(K, transition_matrix, sess_labels, colors, save, threshold=0.15):
    """
    Use pyvis.network to visualize the transition graph
    Inputs:
    - K (dtype: int): number of states
    - transition_matrix (dtype: np.ndarray): the transition matrix
    - sess_labels (dtype: np.ndarray): A single array of session labels to calculate percent of time in each state
    - colors (dtype: list): list of colors
    - save (dtype: str): path to save html file
    - threshold (dtype: float): threshold of [transition/(all transitions from that state except to self)] that determines whether or not to show the transition in the graph
    """
    percents = []
    for i in range(K):
        percents.append(np.sum(sess_labels==i)/sess_labels.shape[0])
    net = Network(directed=True, notebook=True)
    net.add_nodes(range(K), value=percents, color=colors)
    for source in range(K):
        for to in range(K):
            if source != to:
                value = transition_matrix[source, to]
                if transition_matrix[source, to]/(1 - transition_matrix[source,source]) > threshold:
                    net.add_edge(source, to, value=value, title=transition_matrix[source, to], arrow_strikethrough=False)
    net.show(save)
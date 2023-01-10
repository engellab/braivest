import numpy as np
import sys
from braivest.model.emgVAE import emgVAE
from ssm.hmm import MultiHMM, HMM
import ssm
import matplotlib.pyplot as plt
from pyvis.network import Network
import seaborn as sns

def hmm_cross_val(clusters, datasets, n_repeats=3, ind_mask=None):
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

def get_hmm_labels(hmm, encodings_list):
    sess_labels = []
    for split in encodings_list:
        sess_labels.append(hmm.most_likely_states(split))
    return sess_labels

def plot_state_durations(hmm, encodings_list, ordering=None, color_list=None, binwidth=0.4, kde_kws=None):
    sess_labels = get_hmm_labels(hmm, encodings_list)
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

def plot_transition_graph(hmm, sess_labels, colors, save, threshold=0.15):
    percents = []
    for i in range(hmm.K):
        percents.append(np.sum(sess_labels==i)/sess_labels.shape[0])
    net = Network(directed=True, notebook=True)
    net.add_nodes(range(hmm.K), value=percents, color=colors)
    for source in range(hmm.K):
        for to in range(hmm.K):
            if source != to:
                value = hmm.transitions.transition_matrix[source, to]
                if hmm.transitions.transition_matrix[source, to]/(1 - hmm.transitions.transition_matrix[source,source]) > threshold:
                    net.add_edge(source, to, value=value, title=hmm.transitions.transition_matrix[source, to], arrow_strikethrough=False)
    net.show(save)
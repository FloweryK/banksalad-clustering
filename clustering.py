import numpy as np
from itertools import repeat
from multiprocessing import Pool
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans


def __job__kmeans_inertia(k, X):
    return KMeans(n_clusters=k, tol=1e-7).fit(X).inertia_


def find_knee(X, N, trials):
    # investigate sequence, d-sequence statistically
    seq = [Pool(processes=4).starmap(__job__kmeans_inertia, zip(range(1, N), repeat(X))) for _ in range(trials)]
    seq = np.array(seq)[:, 1:]
    dseq = abs(seq[:, 1:] - seq[:, :-1]) / (seq[:, :-1])
    seq = seq.mean(axis=0)
    dseq = dseq.mean(axis=0)

    # find the first knee
    # TODO: add minima restrictions to aviod relatively 'weak' minima.
    minima = argrelextrema(dseq, np.less)[0]
    knee = minima[0] + 1

    return seq, dseq, knee


def find_cluster_labels(X, n_clusters):
    # get labels
    labels = KMeans(n_clusters=n_clusters, tol=1e-7).fit(X).labels_.tolist()

    # re-label by cluster element count
    count = sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True)
    conversion = {old: i for i, (old, _) in enumerate(count)}
    labels = [conversion[label] for label in labels]
    return labels
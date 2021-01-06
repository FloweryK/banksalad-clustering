import numpy as np
from itertools import repeat
from multiprocessing import Pool
from sklearn.cluster import KMeans


def __job__kmeans_inertia(k, X):
    kmeans = KMeans(n_clusters=k)
    inertia = kmeans.fit(X).inertia_
    return inertia


def scan_sequence(X, trials, N=15):
    seq = np.array([Pool(processes=4).starmap(__job__kmeans_inertia, zip(range(1, N), repeat(X))) for _ in range(trials)])
    return seq


def find_optimal_K(seq):
    # investigate sequence, d-sequence statistically
    # seq: k -> [1, 2, 3, 4, ..., N]
    # dseq: k -> [1, 2, 3, 4, ..., N-1]
    # phi: k -> [1, 2, 3, 4, ..., N-2]
    # TODO: add minima restrictions to aviod relatively 'weak' minima.
    # TODO: where to locate mean?
    dseq = abs(seq[:, 1:] - seq[:, :-1]) / (seq[:, :-1])
    dseq = dseq.mean(axis=0)
    phi = dseq[1:] - dseq[:-1]
    knees = [i for i, boolean in enumerate(phi > 0) if boolean]
    knee = knees[0] + 1

    return knee


def find_cluster_labels(X, K):
    # get labels
    labels = KMeans(n_clusters=K).fit(X).labels_.tolist()

    # re-label by cluster element count
    count = sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True)
    conversion = {old: i for i, (old, _) in enumerate(count)}
    labels = [conversion[label] for label in labels]
    return labels
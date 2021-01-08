import random
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


def __job__kmeans_inertia(k, X):
    kmeans = KMeans(n_clusters=k, random_state=random.randint(0, 1000))
    inertia = kmeans.fit(X).inertia_
    return inertia


def scan_sequence(X, trials, N=15):
    seq = np.array([Pool(processes=4).starmap(__job__kmeans_inertia, zip(range(1, N), repeat(X))) for _ in range(trials)])
    return seq


def scan_scores(X, k_values):
    return [KMeans(n_clusters=k, random_state=random.randint(0, 1000)).fit(X).inertia_ for k in k_values]


def find_knee(x, y):
    # rough implementation of https://raghavan.usc.edu//papers/kneedle-simplex11.pdf
    x_norm = (x - min(x)) / (max(x) - min(x))
    y_norm = (max(y) - y) / (max(y) - min(y))
    y_distance = y_norm - x_norm
    knee_index = argrelextrema(y_distance, np.greater)[0]
    knee = y[knee_index]

    '''
    dseq = abs(seq[:, 1:] - seq[:, :-1]) / (seq[:, :-1])
    dseq = dseq.mean(axis=0)
    phi = dseq[1:] - dseq[:-1]
    knees = [i for i, boolean in enumerate(phi > 0) if boolean]
    knee = knees[0] + 1'''

    return knee


def find_kmeans_labels(X, K):
    # get labels
    labels = KMeans(n_clusters=K).fit(X).labels_.tolist()

    # re-label by cluster element count
    count = sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True)
    conversion = {old: i for i, (old, _) in enumerate(count)}
    labels = [conversion[label] for label in labels]
    return labels
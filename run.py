import os
import pandas as pd
from funcs import *
from config import *
from clustering import *
from visualization import *
from data_generation import *


def run(banksalad_path, norm, metric, trials):
    # SAVE PATH
    os.makedirs('images/', exist_ok=True)
    save_as = 'images/%s_norm=%s_trials=%s' % (metric, norm, trials)

    # DATA LOADING
    # TODO: argument -> freq
    df = load_banksalad_as_df(banksalad_path, freq='W')

    # DATA GENERATION
    # TODO: argument -> DF_ORG_PATH
    # TODO: argument -> mul
    # TODO: argument -> if generate data or not
    # df = pd.concat([df, generate_from_clusters(DF_ORG_PATH, mul=10)])

    # DATA PREPROCESSING
    # TODO: mean translate
    if norm:
        df = normalize(df)

    # CLUSTERING K SELECTION
    # TODO: scanning range (currently hard-coded)
    # TODO: Kmeans random seed problem?
    seq = scan_sequence(X=convert_metric(df, metric), trials=trials)
    K = find_optimal_K(seq)
    labels = find_cluster_labels(X=convert_metric(df, metric), K=K)

    # VISUALIZE ELBOW METHOD
    visualize_elbow_method(seq, K, save_as=save_as + '_elbow.jpg')
    visualize_heatmap(df, labels, metric, save_as=save_as + '_heatmap.jpg')
    visualize_clusters(df, labels, K, save_as=save_as + '_clusters.jpg')
    visualize_in_2D(df, labels, save_as=save_as + '_PCA.jpg')


if __name__ == '__main__':
    run(banksalad_path=PATH,
        norm=NORM,
        metric=METRIC,
        trials=TRIALS)

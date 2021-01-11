import os
import pandas as pd
from funcs import *
from config import *
from clustering import *
from visualization import *
from data_generation import *
from multiprocessing import Pool
from itertools import repeat


def run(banksalad_path, generate, norm, metric, trials, save):
    # SAVE PATH
    os.makedirs('images/', exist_ok=True)
    save_as = 'images/%s_gen=%s_norm=%s_trials=%s' % (metric, generate, norm, trials)

    # DATA LOADING
    # TODO: argument -> freq
    df = load_banksalad_as_df(banksalad_path, freq='W')

    # DATA GENERATION
    # TODO: argument -> mul
    if generate:
        # df_gen = generate_from_clusters(load_path=SAVE_PATH, mul=10)
        df_gen = generate_from_patterns(patterns=PATTERNS, columns=df.columns)
        # df = pd.concat([df, df_gen])
        df = df_gen

    # DATA PREPROCESSING
    # TODO: mean translate
    if norm:
        df = normalize(df)

    # CLUSTERING K SELECTION
    # TODO: scanning range (currently hard-coded)
    # TODO: multiprocessing
    # clustering, visualizing as a single class
    X = convert_metric(df, metric)
    k_values = np.arange(1, 16)

    seqs = np.array([[KMeans(n_clusters=k, random_state=random.randint(0, 1000)).fit(X).inertia_ for k in k_values] for _ in range(trials)])
    seqs_norm = ((np.max(seqs, axis=1) - seqs.T) / (np.max(seqs, axis=1) - np.min(seqs, axis=1))).T.mean(axis=0)
    k_norm = (k_values - min(k_values)) / (max(k_values) - min(k_values))
    seqs_distance = seqs_norm - k_norm
    k_distance = k_norm
    knee_index = argrelextrema(seqs_distance, np.greater)[0][0]
    knee = k_values[knee_index]
    plt.plot(k_values, seqs_norm, label='sequences')
    plt.plot(k_values, seqs_distance, label='seqeunce distance')
    plt.grid()
    plt.legend()
    plt.show()

    labels = find_kmeans_labels(X=convert_metric(df, metric), K=knee)

    # VISUALIZE ELBOW METHOD
    # visualize_elbow_method(seq, knee, save_as=save_as + '_elbow.jpg')
    visualize_heatmap(df, labels, metric, save_as=save_as + '_heatmap.jpg')
    visualize_clusters(df, labels, knee, save_as=save_as + '_clusters.jpg')
    visualize_in_2D(df, labels, save_as=save_as + '_PCA.jpg')

    # LABEL SAVING FOR DATA GENERATION
    if save:
        df_label = df.copy()
        df_label['label'] = labels
        df_label.to_csv(SAVE_PATH, encoding='utf-8-sig')


if __name__ == '__main__':
    run(banksalad_path=LOAD_PATH,
        generate=GENERATE,
        norm=NORM,
        metric=METRIC,
        trials=TRIALS,
        save=SAVE)

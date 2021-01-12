import os
import pandas as pd
from funcs import *
from config import *
from clustering import *
from data_generation import *


def run(load_path, metric, trials):
    os.makedirs('images/', exist_ok=True)
    save_as = 'images/metric=%s_trials=%02i' % (metric, trials)

    # LOAD DATA
    df = load_banksalad_as_df(load_path, freq='W')

    # PREPROCESS DATA
    df_norm = normalize_df(df)

    # FIND KNEE
    elbowMethod = ElbowMethod(df=df_norm)
    elbowMethod.find_knee(metric=metric, K=np.arange(1, 16), trials=trials)
    elbowMethod.visualize_elbow_method(save_as=save_as + '_elbowmethod.jpg')
    knee = elbowMethod.knee

    # PERFORM KMEANS AT KNEE
    X = convert_metric(df=df_norm, metric=metric)
    labels = KMeans(n_clusters=knee).fit(X).labels_.tolist()

    # VISUALIZE RESULTS
    summarizer = Summarizer(df=df, labels=labels)
    summarizer.visualize_heatmap(save_as=save_as + '_heatmap.jpg', metric=metric)
    summarizer.visualize_barchart(save_as=save_as + '_barchart.jpg')
    summarizer.visualize_radarchart(save_as=save_as + '_radarchart.jpg')


if __name__ == '__main__':
    run(load_path=LOAD_PATH,
        metric=METRIC,
        trials=TRIALS)

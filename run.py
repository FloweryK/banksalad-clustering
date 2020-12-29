import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from clustering import *
from visualization import *
from funcs import *

# pandas max display options (only for code testing and monitoring)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# matplotlib korean option
# You have to manually install proper korean font if you want to use in korean.
# Otherwise, all korean characters might not appear properly.
mpl.rcParams['font.family'] = 'AppleSDGothicNeoM00'
mpl.rcParams['axes.unicode_minus'] = False


def run(path, measure, norm, mean, N, trials, save_as):
    # load banksalad data
    df = load_banksalad_as_df(path)

    # normalize
    df = normalize(df, norm=norm, mean=mean)

    # visualize
    fig = plt.figure(figsize=(15, 10))

    # perform elbow method
    seq, dseq, knee = find_knee(X=convert_metric(df, measure), trials=trials, N=N)

    # perform clustering at knee
    df['label'] = find_cluster_labels(X=convert_metric(df, measure), n_clusters=knee)

    # visualize elbow method
    visualize_elbow_method(fig=fig, position=(3, 3, 1), seq=seq, dseq=dseq, knee=knee)

    # visualize before & after clustering heatmap
    visualize_heatmap(fig=fig, position=(3, 3, 2), hm=convert_metric(df.reset_index().set_index(['label', '날짜']).sort_index(axis=0, level=1), measure))
    visualize_heatmap(fig=fig, position=(3, 3, 3), hm=convert_metric(df.reset_index().set_index(['label', '날짜']).sort_index(axis=0, level=0), measure), colorbar=True)

    # visualize grouped bar chart
    for label, group_label in df.groupby('label'):
        group_label = group_label.drop(columns=['label'])
        group_label = group_label.div(group_label.sum(axis=1), axis=0)
        num_clusters = df['label'].nunique()
        visualize_bar_chart(fig=fig, position=(3, num_clusters, 1+1*num_clusters+label), group=group_label, legend=(label == num_clusters-1))
        visualize_radar_chart(fig=fig, position=(3, num_clusters, 1+2*num_clusters+label), group=group_label)

    # show plot
    plt.tight_layout()
    plt.show()
    plt.savefig(save_as)
    plt.close()


def get_arguments():
    # Argument configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # args = get_arguments()
    # path = args.path
    path = 'src/2019-12-18~2020-12-18.xlsx'
    norm = True
    mean = False
    measure = 'cosine'
    N = 15
    trials = 1

    for measure in ['cosine']:
        for norm in [True]:
            run(path=path,
                norm=norm,
                mean=mean,
                measure=measure,
                N=N,
                trials=trials,
                save_as='measure=%s_norm=%s_mean=%s.jpg' % (measure, str(norm), str(mean)))

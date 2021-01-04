import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from funcs import *
from clustering import *
from visualization import *

# pandas max display options (only for code testing and monitoring)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# matplotlib korean option
# You have to manually install proper korean font if you want to use in korean.
# Otherwise, all korean characters might not appear properly.
mpl.rcParams['font.family'] = 'AppleSDGothicNeoM00'
mpl.rcParams['axes.unicode_minus'] = False


def run(path, freq, measure, norm, mean, N, trials):
    # save directory
    save_as = '%s_freq=%s_norm=%s_mean=%s_trial=%02i' % (measure, freq, str(norm), str(mean), trials)

    # load banksalad data
    df = load_banksalad_as_df(path=path,
                              freq=freq)

    # prepare original data
    # TODO: discard dependency of this line
    df_org = df

    # normalize
    df = normalize(df=df,
                   norm=norm,
                   mean=mean)

    # clustering performance
    fig1 = plt.figure(figsize=(15, 10))

    # perform elbow method
    seq, dseq, knee = find_knee(X=convert_metric(df, measure),
                                trials=trials,
                                N=N)

    # perform clustering at knee
    df['label'] = find_cluster_labels(X=convert_metric(df, measure),
                                      n_clusters=knee)

    # visualize elbow method
    visualize_elbow_method(fig=fig1,
                           position=(2, 2, (1, 2)),
                           seq=seq,
                           dseq=dseq,
                           knee=knee)

    # visualize before & after clustering heatmap
    visualize_heatmap(fig=fig1,
                      position=(2, 2, 3),
                      title='before clustering (%s)' % measure,
                      # hm=df.drop(columns=['label']).T.corr(),
                      hm=convert_metric(df.drop(columns=['label']), measure))
    visualize_heatmap(fig=fig1,
                      position=(2, 2, 4),
                      colorbar=True,
                      title='after clustering (%s)' % measure,
                      # hm=df.sort_values('label').drop(columns=['label']).T.corr())
                      hm=convert_metric(df.sort_values('label').drop(columns=['label']), measure),)

    plt.tight_layout()
    plt.savefig(save_as + '_clustering.jpg')
    plt.close()

    # clustering results
    n_clusters = df['label'].nunique()
    df_org['label'] = df['label']
    fig2 = plt.figure(figsize=(3*n_clusters, 10))

    # visualize grouped bar chart
    # use original df only
    for label, group_label in df_org.groupby('label'):
        group_label = group_label.drop(columns=['label'])
        group_label = group_label.div(group_label.sum(axis=1), axis=0)

        visualize_bar_chart(fig=fig2,
                            position=(2, n_clusters, 1+0*n_clusters+label),
                            group=group_label,
                            title='#%i' % label,
                            legend=(label == n_clusters - 1))
        visualize_radar_chart(fig=fig2,
                              position=(2, n_clusters, 1+1*n_clusters+label),
                              group=group_label)

    plt.tight_layout()
    plt.savefig(save_as + '_grouped.jpg')
    plt.close()

    # visualize cluster result in 2D
    fig3 = plt.figure(figsize=(10, 10))
    visualize_in_2D(fig=fig3,
                    df=df.drop(columns=['label']),
                    labels=df['label'].reset_index().drop(columns=['날짜']))

    plt.tight_layout()
    plt.savefig(save_as + '_2d.jpg')
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
    freq = 'W'
    norm = True
    mean = False
    measure = 'cosine'
    N = 15
    trials = 1

    run(path=path,
        freq=freq,
        norm=norm,
        mean=mean,
        measure=measure,
        N=N,
        trials=trials)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from funcs import *
from config import *
from clustering import *
from visualization import *
from data_generation import *

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
    os.makedirs('images/', exist_ok=True)
    save_as = 'images/%s_freq=%s_norm=%s_mean=%s_trial=%02i' % (measure, freq, str(norm), str(mean), trials)

    # load banksalad data
    df = load_banksalad_as_df(path=path,
                              freq=freq)

    # concat loaded & generated data
    df = pd.concat([df, generate_from_clusters(DF_ORG_PATH, n_samples=200)])
    df_tmp = df

    # normalize
    df = normalize(df=df, norm=norm, mean=mean)

    # perform elbow method and cluster at knee
    seq, dseq, n_clusters = find_knee(X=convert_metric(df, measure),
                                      trials=trials,
                                      N=N)
    df['label'] = find_cluster_labels(X=convert_metric(df, measure), n_clusters=n_clusters)
    df_tmp['label'] = df['label']

    # visualize elbow method, before & after clustering heatmap
    fig1 = plt.figure(figsize=(15, 10))
    hm_before = convert_metric(df.drop(columns=['label']), measure)
    hm_after = convert_metric(df.sort_values('label').drop(columns=['label']), measure)
    position_elbow = (2, 2, (1, 2))
    position_hm_before = (2, 2, 3)
    position_hm_after = (2, 2, 4)

    visualize_elbow_method(fig=fig1,
                           position=position_elbow,
                           seq=seq,
                           dseq=dseq,
                           knee=n_clusters)
    visualize_heatmap(fig=fig1,
                      hm=hm_before,
                      position=position_hm_before,
                      title='before clustering (%s)' % measure)
    visualize_heatmap(fig=fig1,
                      position=position_hm_after,
                      hm=hm_after,
                      title='after clustering (%s)' % measure,
                      colorbar=True)
    plt.tight_layout()
    plt.savefig(save_as + '_clustering.jpg')
    plt.close()

    # copy clustering results to df_org
    # TODO how can i handle this part?
    # df_org['label'] = df['label']
    # df_org.to_csv(DF_ORG_PATH, encoding='utf-8-sig')

    # visualize grouped bar chart (use df_org only)
    # TODO cleaning
    fig2 = plt.figure(figsize=(3*n_clusters, 10))

    # TODO cleaning df_tmp
    for label, group_label in df_tmp.groupby('label'):
        group_label = group_label.drop(columns=['label'])
        group_label = group_label.div(group_label.sum(axis=1), axis=0)
        position_bar = (2, n_clusters, 1+label)
        position_radar = (2, n_clusters, 1+1*n_clusters+label)

        # TODO cleaning: group
        visualize_bar_chart(fig=fig2,
                            position=position_bar,
                            group=group_label,
                            title='#%i' % label,
                            legend=(label == n_clusters - 1))
        visualize_radar_chart(fig=fig2,
                              position=position_radar,
                              group=group_label)
    plt.tight_layout()
    plt.savefig(save_as + '_grouped.jpg')
    plt.close()

    # visualize cluster result in 2D
    fig3 = plt.figure(figsize=(10, 10))

    # TODO cleaning df
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
    # PATH = args.path

    run(path=PATH,
        freq=FREQ,
        norm=NORM,
        mean=MEAN,
        measure=MEASURE,
        N=N,
        trials=TRIALS)

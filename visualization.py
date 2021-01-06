import numpy as np
import pandas as pd
import seaborn as sns
from math import pi
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from funcs import convert_metric

import sys
np.set_printoptions(threshold=sys.maxsize)


# matplotlib korean option
# You have to manually install proper korean font if you want to use in korean.
# Otherwise, all korean characters might not appear properly.
mpl.rcParams['font.family'] = 'AppleSDGothicNeoM00'
mpl.rcParams['axes.unicode_minus'] = False


def visualize_elbow_method(seq, K):
    dseq = abs(seq[:, 1:] - seq[:, :-1]) / (seq[:, :-1])
    seq_std = seq.std(axis=0)
    seq_mean = seq.mean(axis=0)
    dseq_std = dseq.std(axis=0)
    dseq_mean = dseq.mean(axis=0)
    x_seq = np.arange(1, len(seq_mean) + 1)
    x_dseq = np.arange(1, len(dseq_mean) + 1)

    # prepare axis
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()

    # plot
    ax1.errorbar(x_seq, seq_mean, yerr=seq_std, color='red')
    ax2.errorbar(x_dseq, dseq_mean, yerr=dseq_std, color='blue')
    ax2.text(K, dseq_mean[K - 1], 'knee: %i' % K, size=13, horizontalalignment='center', verticalalignment='top')

    # axis settings
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sum_of_squared_distances')
    ax2.set_ylabel('dJ')
    ax1.set_xticks(x_seq)
    ax1.set_title('Elbow Method For Optimal k')
    ax1.grid(which='both')

    # show
    plt.tight_layout()
    plt.show()


def visualize_heatmap(df, labels, metric):
    df_sorted = df.copy()
    df_sorted['label'] = labels
    df_sorted = df_sorted.sort_values('label').drop(columns=['label'])
    hm_before = convert_metric(df, metric)
    hm_after = convert_metric(df_sorted, metric)

    # prepare axis
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # plot
    p1 = ax1.imshow(hm_before, cmap='viridis_r')
    p2 = ax2.imshow(hm_after, cmap='viridis_r')

    # axis settings
    cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(p2, cax=cax2)
    cax1.axis('off')
    ax1.set_xticks(np.arange(len(hm_before)))
    ax1.set_yticks(np.arange(len(hm_before)))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_title(metric + ' distance (before clustering)')
    ax2.set_xticks(np.arange(len(hm_before)))
    ax2.set_yticks(np.arange(len(hm_before)))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_title(metric + ' distance (before clustering)')

    # show
    plt.tight_layout()
    plt.show()


def visualize_clusters(df, labels, K):
    df_labeled = df.copy()
    df_labeled['label'] = labels

    # bar
    colors = sns.color_palette('hls', len(df_labeled.columns))

    # radar
    categories = df.columns
    num_categories = len(categories)
    angles = [x / float(num_categories) * (2 * pi) for x in range(num_categories)]
    angles += angles[:1]
    angles = np.array(angles)
    r_max = df.to_numpy().max()

    # prepare axis
    fig = plt.figure(figsize=(3*K, 10))

    # plot
    for i, (label, group) in enumerate(df_labeled.groupby(['label'])):
        # drop label. we will not use it anymore
        group = group.drop(columns=['label'])

        # bar chart
        ax_bar = fig.add_subplot(2, K, 1+i)

        # plot
        bottom = np.zeros(len(group.index))
        for category, color in zip(group, colors):
            ax_bar.bar(group.index, group[category].to_numpy(), bottom=bottom, label=category, color=color)
            bottom += group[category].to_numpy()

        # axis settings
        ax_bar.set_xticklabels([], fontsize=6, rotation=90)
        ax_bar.set_yticklabels([])
        ax_bar.set_ylim([0, 1])
        ax_bar.set_title('#' + str(label))
        if i == K-1:
            fig.subplots_adjust(right=0.7)
            ax_bar.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # radar chart
        ax_radar = fig.add_subplot(2, K, 1+K+i, polar=True)

        # group plot
        for j, row in group.iterrows():
            data = group.loc[j].to_list()
            data += data[:1]
            ax_radar.plot(angles, data, linewidth=0.5, linestyle='solid', color='grey', alpha=0.6)

        # mean data plot
        data = group.mean(axis=0).to_list()
        data += data[:1]
        ax_radar.plot(angles, data, linewidth=2, linestyle='solid')
        ax_radar.fill(angles, data, alpha=0.4)

        # axis settings
        ax_radar.set_theta_offset(pi / 2)
        ax_radar.set_theta_direction(-1)
        ax_radar.set_rlim([0, r_max])
        ax_radar.set_xticklabels([])
        ax_radar.set_yticklabels([])
        ax_radar.set_thetagrids(angles[:-1] * (180 / pi), labels=categories, fontsize=8)

    # show
    plt.tight_layout()
    plt.show()


def visualize_in_2D(df, labels):
    X = PCA(n_components=2).fit_transform(df)
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['label'] = labels

    # plot
    sns.scatterplot(data=X, x='x', y='y', hue='label', style='label', s=300)

    # axis settings
    plt.title('clustering results (PCA reduced)')

    # show
    plt.tight_layout()
    plt.show()


import random
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import pi
from itertools import repeat
from multiprocessing import Pool
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
from mpl_toolkits.axes_grid1 import make_axes_locatable
from funcs import *


# matplotlib korean option
# You have to manually install proper korean font if you want to use in korean.
# Otherwise, all korean characters might not appear properly.
mpl.rcParams['font.family'] = 'AppleSDGothicNeoM00'
mpl.rcParams['axes.unicode_minus'] = False


class ElbowMethod:
    def __init__(self, df):
        self.df = df

        self.seqs = None
        self.seqs_distance = None
        self.knee = None

    @staticmethod
    def normalize_min_max(x, reverse=False):
        # x should be in numpy array
        if reverse:
            return (max(x) - x) / (max(x) - min(x))
        else:
            return (x - min(x)) / (max(x) - min(x))

    @staticmethod
    def get_kmeans_inertia(X, k):
        return KMeans(n_clusters=k, random_state=random.randint(0, 1000)).fit(X).inertia_

    def find_knee(self, metric, K=np.arange(1, 16), trials=2):
        X = convert_metric(self.df, metric)

        seqs = np.array([Pool(processes=6).starmap(self.get_kmeans_inertia, zip(repeat(X), K)) for _ in range(trials)])

        # normalize
        K_norm = self.normalize_min_max(K)
        seqs_norm = np.array([self.normalize_min_max(x, reverse=True) for x in seqs])

        # distance
        seqs_distance = seqs_norm - K_norm

        # find knee with seqs_distance mean
        knee = int(K[argrelextrema(seqs_distance.mean(axis=0), np.greater)][0])

        # save
        self.K = K
        self.knee = knee
        self.seqs = seqs
        self.seqs_distance = seqs_distance

    def visualize_elbow_method(self, save_as):
        seqs_std = self.seqs.std(axis=0)
        seqs_mean = self.seqs.mean(axis=0)
        seqs_norm_std = self.seqs_distance.std(axis=0)
        seqs_norm_mean = self.seqs_distance.mean(axis=0)

        # prepare axis
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot()
        ax2 = ax1.twinx()

        # plot
        ax1_color = 'red'
        ax2_color = 'blue'
        ax1.errorbar(self.K, seqs_mean, yerr=seqs_std, label='J', color=ax1_color, marker='x')
        ax2.errorbar(self.K, seqs_norm_mean, yerr=seqs_norm_std, label='dJ', color=ax2_color, marker='x')
        ax2.text(self.knee, seqs_norm_mean[self.knee-1]-0.01, 'knee: %i' % self.knee, size=13, horizontalalignment='center', verticalalignment='top')

        # axis settings
        ax1.set_xlabel('K')
        ax1.set_ylabel('Sum_of_squared_distances', color=ax1_color)
        ax2.set_ylabel('dJ', color=ax2_color)
        ax1.set_xticks(self.K)
        ax1.grid(which='both')
        ax1.tick_params(axis='y', labelcolor=ax1_color)
        ax2.tick_params(axis='y', labelcolor=ax2_color)
        ax1.set_title('Elbow Method For Optimal k')

        # show
        plt.tight_layout()
        plt.savefig(save_as)
        plt.close()


class Summarizer:
    def __init__(self, df, labels):
        self.df = df
        self.labels = self.sort_label_number(labels)
        self.n_clusters = len(set(labels))

    @staticmethod
    def sort_label_number(labels):
        count = sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True)
        conversion = {old: i for i, (old, _) in enumerate(count)}
        labels = [conversion[label] for label in labels]
        return labels

    def visualize_heatmap(self, metric, save_as):
        df_sorted = self.df.copy()
        df_sorted['label'] = self.labels
        df_sorted = df_sorted.sort_values('label').drop(columns=['label'])
        hm_before = convert_metric(self.df, metric)
        hm_after = convert_metric(df_sorted, metric)
        contour_pos = [self.labels.count(label) for label in sorted(list(set(self.labels)))]
        contour_pos = [sum(contour_pos[:i]) for i in range(len(contour_pos))]

        # prepare axis
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # plot
        p1 = ax1.imshow(hm_before, cmap='viridis_r')
        p2 = ax2.imshow(hm_after, cmap='viridis_r')
        for i, pos in enumerate(contour_pos):
            ax2.axvline(pos - 0.5, color='white')
            ax2.axhline(pos - 0.5, color='white')
            ax2.text(pos + 1, pos + 1, '#' + str(i),
                     size=13,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='square', alpha=0.7, ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8)))

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
        plt.savefig(save_as)
        plt.close()

    def visualize_barchart(self, save_as):
        df_labeled = self.df.copy()
        df_labeled = normalize_df(df_labeled)
        df_labeled['label'] = self.labels

        # bar
        colors = sns.color_palette('hls', len(df_labeled.columns))

        # prepare axis
        fig = plt.figure(figsize=(3 * self.n_clusters, 8))

        # plot
        for i, (label, group) in enumerate(df_labeled.groupby(['label'])):
            # drop label. we will not use it anymore
            group = group.drop(columns=['label'])

            # bar chart
            ax = fig.add_subplot(1, self.n_clusters, 1 + i)

            # plot
            bottom = np.zeros(len(group.index))
            for category, color in zip(group, colors):
                ax.bar(group.index, group[category].to_numpy(), bottom=bottom, label=category, color=color)
                bottom += group[category].to_numpy()

            # axis settings
            ax.set_xticklabels([], fontsize=6, rotation=90)
            ax.set_yticklabels([])
            ax.set_ylim([0, 1])
            ax.set_title('#' + str(label))
            if i == self.n_clusters - 1:
                fig.subplots_adjust(right=0.7)
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # show
        plt.tight_layout()
        plt.savefig(save_as)
        plt.close()

    def visualize_radarchart(self, save_as):
        df_labeled = self.df.copy()
        df_labeled = normalize_df(df_labeled)
        df_labeled['label'] = self.labels

        # radar
        categories = self.df.columns
        num_categories = len(categories)
        angles = [x / float(num_categories) * (2 * pi) for x in range(num_categories)]
        angles += angles[:1]
        angles = np.array(angles)
        # r_max = self.df.to_numpy().max()

        # prepare axis
        fig = plt.figure(figsize=(3 * self.n_clusters, 4))

        # plot
        for i, (label, group) in enumerate(df_labeled.groupby(['label'])):
            # drop label. we will not use it anymore
            group = group.drop(columns=['label'])

            # radar chart
            ax = fig.add_subplot(1, self.n_clusters, 1 + i, polar=True)

            # group plot
            for j, row in group.iterrows():
                data = group.loc[j].to_list()
                data += data[:1]
                ax.plot(angles, data, linewidth=0.5, linestyle='solid', color='grey', alpha=0.6)

            # mean data plot
            data = group.mean(axis=0).to_list()
            data += data[:1]
            ax.plot(angles, data, linewidth=2, linestyle='solid')
            ax.fill(angles, data, alpha=0.4)

            # axis settings
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)
            ax.set_rlim([0, 1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_thetagrids(angles[:-1] * (180 / pi), labels=categories, fontsize=8)
            ax.set_title('#' + str(label))

        # show
        plt.tight_layout()
        plt.savefig(save_as)
        plt.close()



import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams['font.family'] = 'AppleSDGothicNeoM00'



class Summary:
    def __init__(self, path, norm=False, mean=False, figsize=(10, 15)):
        self.df = self.load_data(path, norm, mean)
        self.fig = plt.figure(figsize=figsize)

    def load_data(self, path, norm, mean):
        # load raw data
        raw = pd.read_csv(path, index_col='날짜')
        raw.index = pd.to_datetime(raw.index)
        raw = raw[raw['타입'] == '지출']

        # prerequisites
        categories = sorted(list(set(raw['대분류'])))

        dates = []
        amounts = {category: [] for category in categories}

        # fill data
        for month, group in raw.groupby(pd.Grouper(freq='W')):
            dates.append(month)

            for category in categories:
                if category == 'total':
                    amounts[category].append(-group['금액'].sum())
                else:
                    amounts[category].append(-group[group['대분류'] == category]['금액'].sum())

        # convert to dataframe
        df = pd.DataFrame(amounts, index=dates)
        df.index = df.index.strftime('%Y-%m-%d')
        df = df[df.sum(axis=0).sort_values(ascending=False).index]

        # normalize
        if norm:
            df = df.div(df.sum(axis=1), axis=0)
        if mean:
            df -= df.mean(axis=0)

        return df

    @staticmethod
    def get_positions(df, measure):
        if measure == 'cosine':
            return 1 - cosine_similarity(df, df)
        elif measure == 'euclidean':
            return df
        else:
            raise KeyError('invalid measure options')

    @staticmethod
    def get_distance(df, measure):
        if measure == 'cosine':
            return 1 - cosine_similarity(df, df)
        elif measure == 'euclidean':
            return euclidean_distances(df)
        else:
            raise KeyError('invalid measure options')

    @staticmethod
    def plt_show():
        plt.show()

    @staticmethod
    def plt_savefig(path):
        plt.savefig(path)

    @staticmethod
    def plt_close():
        plt.close()

    def add_knee_finding(self, nrow, row, measure, tol):
        # get positions
        positions = self.get_positions(self.df, measure)

        # make knee data
        K = range(1, len(self.df.index))
        seq = [KMeans(n_clusters=k, tol=tol).fit(positions).inertia_ for k in K]
        dseq = [0] + [abs((seq[j + 1] - seq[j])) / (1e-5 + seq[j]) for j in range(len(seq) - 1)]
        knee = 2
        while knee < len(dseq):
            # hard-coded options for dseq value
            if (dseq[knee-1] > dseq[knee] < dseq[knee+1]) and (dseq[knee] < 0.12):
                break
            else:
                knee += 1
        knee += 1

        # add subplot
        ax = self.fig.add_subplot(nrow, 2, (1 + 2 * row, 2 + 2 * row))
        ax.plot(K, seq, 'bx-')
        ax.set_xlabel('K')
        ax.set_ylabel('Sum_of_squared_distances')
        ax.set_xticks(range(1, len(self.df.index), 2))
        ax.set_title('Elbow Method For Optimal k')
        ax.grid(which='both')
        ax2 = ax.twinx()
        ax2.plot(K, dseq, 'rx-')
        ax2.text(knee-1, dseq[knee-1] - 0.04, 'knee: ' + str(knee), size=13)
        ax2.set_ylabel('dJ')

        return knee

    def add_clustering_result(self, nrow, row, measure, n_clusters, tol):
        # get positions
        positions = self.get_positions(self.df, measure)

        # make clustering data
        cluster_index = {}
        for label, index in zip(KMeans(n_clusters=n_clusters, tol=tol).fit(positions).labels_, self.df.index):
            if label not in cluster_index:
                cluster_index[label] = []
            cluster_index[label].append(index)
        cluster_index = {label: indices for label, indices in enumerate(sorted(list(cluster_index.values()), key=lambda l: len(l), reverse=True))}

        # add subplot
        # heatmap before & after clustering
        ax1 = self.fig.add_subplot(nrow, 2, 1+2*row)
        ax2 = self.fig.add_subplot(nrow, 2, 2+2*row)

        # heatmap before clustering
        im1 = self.get_distance(self.df, measure)
        plt1 = ax1.imshow(im1, cmap='viridis_r')
        cax1 = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
        cax1.axis('off')
        ax1.set_xticks(np.arange(len(self.df.index)))
        ax1.set_yticks(np.arange(len(self.df.index)))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_title(measure + ' distance (before clustering)')

        # heatmap after clustering
        im2 = self.get_distance(pd.DataFrame(self.df, index=[index for label in cluster_index for index in cluster_index[label]]), measure)
        plt2 = ax2.imshow(im2, cmap='viridis_r')
        cax2 = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.05)
        plt.colorbar(plt2, cax=cax2)
        ax2.set_xticks(np.arange(len(self.df.index)))
        ax2.set_yticks(np.arange(len(self.df.index)))
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_title(measure + ' distance (after clustering)')

        # put labels on heatmap
        for label, indices in cluster_index.items():
            x = sum([len(_indices) for _indices in list(cluster_index.values())[:label]]) - 0.5
            ax2.text(x, x, '#' + str(label), verticalalignment='top', horizontalalignment='left')
            ax2.axvline(x, color='white')
            ax2.axhline(x, color='white')

        # grouped bar chart
        for i, (label, indices) in enumerate(cluster_index.items()):
            grouped_df = self.df.loc[indices]
            grouped_df = grouped_df.div(grouped_df.sum(axis=1), axis=0)

            ax3 = self.fig.add_subplot(nrow, n_clusters, 1+n_clusters*(row+1)+label)

            bottom = np.zeros(len(grouped_df.index))
            for category in grouped_df:
                ax3.bar(grouped_df.index, grouped_df[category].to_numpy(), bottom=bottom, label=category)
                bottom += grouped_df[category].to_numpy()

            if i == len(cluster_index)-1:
                self.fig.subplots_adjust(right=0.7)
                ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax3.set_title('#' + str(label))
            ax3.set_xticklabels(grouped_df.index, fontsize=6, rotation=90)
            ax3.set_yticklabels([])
            ax3.set_ylim([0, 1])


def get_arguments():
    # Argument configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)


    return parser.parse_args()


def run():
    args = get_arguments()
    # path = 'src/2019-12-18~2020-12-18.csv'
    path = args.path
    tol = 1e-7
    mean = False

    for norm in [True, False]:
        for measure in ['cosine', 'euclidean']:
            print(norm, mean, measure)
            summary = Summary(path=path, norm=norm, mean=False, figsize=(10, 15))
            knee = summary.add_knee_finding(nrow=3, row=0, measure=measure, tol=tol)
            summary.add_clustering_result(nrow=3, row=1, measure=measure, n_clusters=knee, tol=tol)
            summary.plt_savefig('measure=%s_norm=%s_mean=%s.jpg' % (str(measure), str(norm), str(mean)))
            summary.plt_close()


if __name__ == '__main__':
    run()

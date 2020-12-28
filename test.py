import argparse
import numpy as np
import pandas as pd
from itertools import repeat
from multiprocessing import Pool
from openpyxl import load_workbook
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# pandas option
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# matplotlib korean option
mpl.rcParams['font.family'] = 'AppleSDGothicNeoM00'
mpl.rcParams['axes.unicode_minus'] = False


def run(path, measure, norm, mean, save_as):
    # load banksalad data
    df = load_banksalad_as_df(path)

    # normalize
    df = normalize(df, norm=norm, mean=mean)

    # cluster summary
    get_cluster_summary(df, measure=measure, save_as=save_as)


def load_banksalad_as_df(path):
    # open xlsx worksheet
    wb = load_workbook(path)
    ws = wb['가계부 내역']

    # convert worksheet into dataframe
    raw = pd.DataFrame(ws.values)
    raw.columns = raw.loc[0].tolist()
    raw = raw.drop(index=[0])
    raw.index = raw['날짜']
    raw = raw.drop(columns=['날짜'])

    # extract target types
    # TODO: currently using hard-coded type only
    raw = raw[raw['타입'] == '지출']

    # make categorized daily outcome
    # TODO: currently using hard-coded high-hierarchy category only
    df = pd.DataFrame(columns=set(raw['대분류']))

    # TODO: currently using harded-coded grouping options
    for date, group in raw.groupby(pd.Grouper(freq='W')):
        df.loc[date] = {category: abs(group2['금액'].sum()) for category, group2 in group.groupby('대분류')}

    # beautify dataframe
    df = df.fillna(0)
    df = df.sort_index(axis=0)
    df = df.sort_index(axis=1, key=lambda x: df[x].sum(), ascending=False)
    df.index = df.index.strftime('%Y-%m-%d')
    df.index.name = '날짜'

    return df


def normalize(df, norm, mean):
    if norm:
        # TODO: more normalize options (ex. axis=0 dividing, L2 norm)
        df = df.div(df.sum(axis=1), axis=0)
    if mean:
        df -= df.mean(axis=0)

    return df


def __job__kmeans_inertia(k, X):
    return KMeans(n_clusters=k, tol=1e-7).fit(X).inertia_


def get_cluster_summary(df, measure, save_as):
    def find_knee(X, N, trials):
        # investigate sequence, d-sequence statistically
        seq = [Pool(processes=4).starmap(__job__kmeans_inertia, zip(range(1, N), repeat(X))) for _ in range(trials)]
        seq = np.array(seq)[:, 1:]
        dseq = abs(seq[:, 1:] - seq[:, :-1]) / (1e-5 + seq[:, :-1])
        seq = seq.mean(axis=0)
        dseq = dseq.mean(axis=0)

        # find the first knee
        # TODO: add minima restrictions to aviod relatively 'weak' minima.
        minima = argrelextrema(dseq, np.less)[0]
        knee = minima[0] + 1

        return seq, dseq, knee

    def find_knee_labels(X, knee):
        # get labels
        labels = KMeans(n_clusters=knee, tol=1e-7).fit(X).labels_.tolist()

        # re-label by cluster element count
        count = sorted([(label, labels.count(label)) for label in set(labels)], key=lambda x: x[1], reverse=True)
        conversion = {old: i for i, (old, _) in enumerate(count)}
        labels = [conversion[label] for label in labels]
        return labels

    def convert_metric(df, measure):
        if measure == 'cosine':
            return 1 - cosine_similarity(df, df)
        elif measure == 'euclidean':
            return euclidean_distances(df, df)
        else:
            raise KeyError

    # visualize
    fig = plt.figure(figsize=(10, 15))

    # visualize elbow method
    # TODO: currently using hard-coded N and trials
    seq, dseq, knee = find_knee(X=convert_metric(df, measure), N=15, trials=1)

    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax2 = ax1.twinx()
    ax1.plot(np.arange(1, len(seq) + 1), seq, 'bx-')
    ax2.plot(np.arange(1, len(dseq) + 1), dseq, 'rx-')
    ax2.text(knee, dseq[knee - 1], 'knee: %i' % knee, size=13, horizontalalignment='center', verticalalignment='top')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sum_of_squared_distances')
    ax2.set_ylabel('dJ')
    ax1.set_xticks(np.arange(1, len(seq) + 1, 2))
    ax1.set_title('Elbow Method For Optimal k')
    ax1.grid(which='both')

    # visualize heatmap
    df['label'] = find_knee_labels(X=convert_metric(df, measure), knee=knee)
    hm_before = convert_metric(df.reset_index().set_index(['label', '날짜']).sort_index(axis=0, level=1), measure)
    hm_after = convert_metric(df.reset_index().set_index(['label', '날짜']).sort_index(axis=0, level=0), measure)

    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    plt3 = ax3.imshow(hm_before, cmap='viridis_r')
    plt4 = ax4.imshow(hm_after, cmap='viridis_r')
    for i in range(len(df['label'].value_counts())):
        x = df['label'].value_counts()[:i].sum() - 0.5
        ax4.text(x, x, '#' + str(i), verticalalignment='top', horizontalalignment='left')
        ax4.axvline(x, color='white')
        ax4.axhline(x, color='white')
    cax3 = make_axes_locatable(ax3).append_axes("right", size="5%", pad=0.05)
    cax4 = make_axes_locatable(ax4).append_axes("right", size="5%", pad=0.05)
    cax3.axis('off')
    plt.colorbar(plt4, cax=cax4)
    ax3.set_xticks(np.arange(len(hm_before)))
    ax3.set_yticks(np.arange(len(hm_before)))
    ax4.set_xticks(np.arange(len(hm_after)))
    ax4.set_yticks(np.arange(len(hm_after)))
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    # TODO: title set
    ax3.set_title(measure + ' distance (before clustering)')
    ax4.set_title(measure + ' distance (after clustering)')

    # grouped bar chart
    colors = sns.color_palette('hls', len(df.columns)-1)

    for label, group_label in df.groupby('label'):
        group_delabel = group_label.drop(columns=['label'])
        group_delabel = group_delabel.div(group_delabel.sum(axis=1), axis=0)
        bottom = np.zeros(len(group_delabel.index))

        # plot bars
        ax5 = fig.add_subplot(3, df['label'].nunique(), 1 + 2 * df['label'].nunique() + label)
        for i, category in enumerate(group_delabel):
            ax5.bar(group_delabel.index, group_delabel[category].to_numpy(), bottom=bottom, label=category, color=colors[i])
            bottom += group_delabel[category].to_numpy()

        # axis setting
        if label == df['label'].nunique()-1:
            fig.subplots_adjust(right=0.7)
            ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax5.set_title('#' + str(label))
        ax5.set_xticklabels([], fontsize=6, rotation=90)
        ax5.set_yticklabels([])
        ax5.set_ylim([0, 1])

    # show plot
    # plt.tight_layout()
    plt.savefig(save_as)
    plt.close()


def get_arguments():
    # Argument configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    path = args.path
    # path = 'src/2019-12-18~2020-12-18.xlsx'
    norm = True
    mean = False
    measure = 'cosine'

    for measure in ['cosine', 'euclidean']:
        for norm in [True, False]:
            run(path=path, norm=norm, mean=mean, measure=measure, save_as='measure=%s_norm=%s_mean=%s.jpg' % (measure, str(norm), str(mean)))

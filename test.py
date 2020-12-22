import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleSDGothicNeoM00'

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def run():
    # load raw data
    raw = pd.read_csv('src/2019-12-18~2020-12-18.csv', index_col='날짜')
    raw.index = pd.to_datetime(raw.index)
    raw = raw[raw['타입'] == '지출']

    # prerequisites
    categories = sorted(list(set(raw['대분류'])))
    dates = []
    amounts = {
        category: []
        for category in categories
    }

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
    df = df.div(df.sum(axis=1), axis=0)
    df.index = df.index.strftime('%Y-%m-%d')

    # sort category order by sum
    # categories = [category for _, category in sorted(zip(df.sum(axis=0), categories), reverse=True)]

    '''
    # stacked bar plot
    fig, axes = plt.subplots(figsize=(15, 5))
    bottom = np.zeros(len(df.index))
    for category in categories:
        axes.bar(df.index, df[category].to_numpy(), bottom=bottom, label=category)
        bottom += df[category].to_numpy()

    axes.set_ylabel('지출 구성')
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # plt.grid(True, axis='y')
    plt.xticks(df.index, rotation=90)
    plt.title('월 지출')
    plt.tight_layout()
    plt.show()
    '''

    '''
    # simple line plot
    fig, axes = plt.subplots(figsize=(20, 5))
    for i, category in enumerate(categories):
        axes.plot(df3[category], label=category, linestyle='-', linewidth=1., marker='o', markersize=4)

    # axis settings
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    '''

    # cluster number determination
    '''
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    Sequence = []
    K = range(1, len(df.index))
    for k in K:
        km = KMeans(n_clusters=k).fit(df)
        Sequence.append(km.inertia_)

    dSequence = [0] + [abs((Sequence[j+1]-Sequence[j]))/(1e-5 + Sequence[j]) for j in range(len(Sequence)-1)]
    ax1.plot(K, Sequence, 'bx-')
    ax2.plot(K, dSequence, 'rx-')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sum_of_squared_distances')
    ax2.set_ylabel('dJ')
    ax1.set_xticks(range(1, len(df.index), 2))
    ax1.grid(which='both')
    plt.title('Elbow Method For Optimal k')
    plt.tight_layout()
    plt.show()
    '''

    # clustering
    clustered = KMeans(n_clusters=7).fit(df)
    labels = clustered.labels_

    # sort index order by clustering
    clustered_index = [index for _, index in sorted(zip(labels, df.index))]

    # visualization - heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # subplot 0: heatmap before clustering
    axes[0].imshow(euclidean_distances(df, df), cmap='viridis_r')
    axes[0].set_xticks(np.arange(len(df.index)))
    axes[0].set_yticks(np.arange(len(df.index)))
    axes[0].set_xticklabels(df.index, rotation=90, fontsize=7)
    axes[0].set_yticklabels(df.index, fontsize=7)
    axes[0].set_title('cosine sim. heatmap (before clustering)')

    # subplot 1: heatmap after clustering
    df = pd.DataFrame(df, index=clustered_index)
    axes[1].imshow(euclidean_distances(df, df), cmap='viridis_r')
    axes[1].set_xticks(np.arange(len(df.index)))
    axes[1].set_yticks(np.arange(len(df.index)))
    axes[1].set_xticklabels(df.index, rotation=90, fontsize=7)
    axes[1].set_yticklabels(df.index, fontsize=7)
    axes[1].set_title('cosine sim. heatmap (after clustering)')

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()

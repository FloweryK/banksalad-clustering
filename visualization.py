import numpy as np
import pandas as pd
import seaborn as sns
from math import pi
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA


def visualize_elbow_method(fig, position, seq, dseq, knee):
    ax1 = fig.add_subplot(*position)
    ax2 = ax1.twinx()

    # plot
    ax1.plot(np.arange(1, len(seq) + 1), seq, 'bx-')
    ax2.plot(np.arange(1, len(dseq) + 1), dseq, 'rx-')
    ax2.text(knee, dseq[knee - 1], 'knee: %i' % knee, size=13, horizontalalignment='center', verticalalignment='top')

    # axis settings
    ax1.set_xlabel('K')
    ax1.set_ylabel('Sum_of_squared_distances')
    ax2.set_ylabel('dJ')
    ax1.set_xticks(np.arange(1, len(seq) + 1))
    ax1.set_title('Elbow Method For Optimal k')
    ax1.grid(which='both')


def visualize_heatmap(fig, position, hm, title, colorbar=False):
    ax = fig.add_subplot(*position)

    # plot
    p = ax.imshow(hm, cmap='viridis_r')

    # axis settings
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    if colorbar:
        plt.colorbar(p, cax=cax)
    else:
        cax.axis('off')
    ax.set_xticks(np.arange(len(hm)))
    ax.set_yticks(np.arange(len(hm)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)
    # ax.set_title(measure + ' distance (before clustering)')


def visualize_bar_chart(fig, position, group, title, legend=False):
    ax = fig.add_subplot(*position)

    # plot
    bottom = np.zeros(len(group.index))
    colors = sns.color_palette('hls', len(group.columns))

    for i, category in enumerate(group):
        ax.bar(group.index, group[category].to_numpy(), bottom=bottom, label=category, color=colors[i])
        bottom += group[category].to_numpy()

    # axis settings
    # ax.set_title('#' + str(label))
    ax.set_xticklabels([], fontsize=6, rotation=90)
    ax.set_yticklabels([])
    ax.set_ylim([0, 1])
    ax.set_title(title)
    if legend:
        fig.subplots_adjust(right=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def visualize_radar_chart(fig, position, group):
    ax = fig.add_subplot(*position, polar=True)

    categories = group.columns
    num_categories = len(categories)
    angles = [x/float(num_categories)*(2*pi) for x in range(num_categories)]
    angles += angles[:1]
    angles = np.array(angles)

    # group plot
    for i, row in group.iterrows():
        data = group.loc[i].to_list()
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
    ax.set_rlim([0, 0.6])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_thetagrids(angles[:-1]*(180/pi), labels=categories, fontsize=8)


def visualize_in_2D(fig, df, labels):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def get_eclipse_shape(group, color, nst=2, inc=1.2, lw=2):
        cov = np.cov(group['x'], group['y'])
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * nst * np.sqrt(vals)
        center = group.mean(axis=0).values
        ell = patches.Ellipse(center, width=inc * w, height=inc * h, angle=theta, color=color, alpha=0.2, lw=0)
        edge = patches.Ellipse(center, width=inc * w, height=inc * h, angle=theta, edgecolor=color, facecolor='none', lw=lw)
        return ell, edge

    ax = fig.add_subplot()

    # calculate reduced positions
    X = PCA(n_components=2).fit_transform(df)
    X = pd.DataFrame(X, columns=['x', 'y'])
    X['label'] = labels

    # plot
    sns.scatterplot(data=X,
                    x='x',
                    y='y',
                    hue='label',
                    style='label',
                    ax=ax,
                    s=300)

    # axis settings
    ax.set_title('clustering results (PCA reduced)')

    '''
    # color setting
    colors = sns.color_palette('hls', X['label'].nunique())

    for i, (_, group) in enumerate(X.groupby('label')):
        # scatter cluster elements
        ax.scatter(group['x'], group['y'], color=colors[i])

        # add cluster shape as eclipse
        ell, edge = get_eclipse_shape(group, color=colors[i])
        ax.add_artist(ell)
        ax.add_artist(edge)'''



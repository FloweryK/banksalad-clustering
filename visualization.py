import numpy as np
import seaborn as sns
from math import pi
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def visualize_heatmap(fig, position, hm, colorbar=False):
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
    # ax.set_title(measure + ' distance (before clustering)')


def visualize_bar_chart(fig, position, group, legend=False):
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

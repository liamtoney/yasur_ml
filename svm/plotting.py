import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from svm import ALL_DAYS, ALL_STATIONS


def plot_generalization_matrix(scores, fig, ax, colorbar=True, show_stats=True):

    im = ax.imshow(scores, cmap=cc.m_diverging_bwr_20_95_c54_r, vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_DAYS)))
    ax.set_yticks(range(len(ALL_STATIONS)))
    ax.set_xticklabels([d.strftime('%-d\n%B') for d in ALL_DAYS])
    ax.set_yticklabels(ALL_STATIONS)
    ax.set_xlabel('Test day', labelpad=10)
    ax.set_ylabel('Test station', labelpad=7)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Colorbar handling
    if isinstance(colorbar, Axes):
        im_ax = None
        cax = colorbar
    elif colorbar is True:
        im_ax = ax
        cax = None
    else:  # If colorbar is not an Axes instance or True, then no colorbar!
        cax = False
    if cax is not False:
        fig.colorbar(
            im,
            ax=im_ax,
            cax=cax,
            label='Accuracy score',
            ticks=plt.MultipleLocator(0.25),  # So 50% is shown!
            format=PercentFormatter(xmax=1),
        )

    # Add text
    for i in range(len(ALL_STATIONS)):
        for j in range(len(ALL_DAYS)):
            this_score = scores[i, j]
            # Choose the best text color for contrast
            if this_score >= 0.7 or this_score <= 0.3:
                color = 'white'
            else:
                color = 'black'
            ax.text(
                j,  # column = x
                i,  # row = y
                s=f'{this_score * 100:.0f}',
                ha='center',
                va='center',
                color=color,
                fontsize=8,
                alpha=0.5,
            )

    # Show or print stats
    mean = scores.mean()
    std = scores.std()
    if show_stats:
        ax.set_title(f'$\mu$ = {mean:.0%}\n$\sigma$ = {std:.1%}', loc='left')
    else:
        print(f'mean_diag = {mean:.0%}')
        print(f'std_diag = {std:.1%}')

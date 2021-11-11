import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter

from svm import ALL_DAYS, ALL_STATIONS, COLOR_CYCLE


def plot_generalization_matrix(scores, fig, ax, colorbar=True, show_stats=True):
    """Make a plot of a 5 x 6 generalization matrix.

    Args:
        scores (numpy.ndarray): Array (5 rows, 6 columns) of scores
        fig (Figure): Existing Matplotlib figure to plot into
        ax (Axes): Existing Matplotlib axes to plot into
        colorbar (bool or Axes): True to automatically place colorbar, False for no
            colorbar, or place into the provided Axes instance
        show_stats (bool): Toggle showing score mean and standard deviation on the
            figure vs. just printing the info
    """

    im = ax.imshow(scores, cmap='BrBG', vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_DAYS)))
    ax.set_yticks(range(len(ALL_STATIONS)))
    ax.set_xticklabels([d.strftime('%-d\n%B') for d in ALL_DAYS])
    ax.set_yticklabels(ALL_STATIONS)
    ax.set_xlabel('Test day', labelpad=10)
    ax.set_ylabel('Test station', labelpad=7)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Colorbar handling
    im_ax = None
    cax = None
    if isinstance(colorbar, Axes):
        cax = colorbar
    elif colorbar is True:
        im_ax = ax
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


def plot_path_effect_matrix(
    scores, fig, ax, day, colorbar=True, show_stats=True, diagonal_metrics=True
):
    """Make a plot of a 5 x 5 path effect matrix.

    Args:
        scores (numpy.ndarray): Array (5 rows, 5 columns) of scores
        fig (Figure): Existing Matplotlib figure to plot into
        ax (Axes): Existing Matplotlib axes to plot into
        day (UTCDateTime): Testing day for the scores (will be formated as a date)
        colorbar (bool or Axes): True to automatically place colorbar, False for no
            colorbar, or place into the provided Axes instance (if Axes, orientation
            is set to horizontal!)
        show_stats (bool): Toggle showing score mean and standard deviation and testing
            day on the figure vs. just printing the info
        diagonal_metrics (bool): If True, plot mean and standard deviation for diagonal;
            otherwise, plot for entire 5 x 5 matrix
    """

    im = ax.imshow(scores, cmap='Greys', vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_STATIONS)))
    ax.set_yticks(range(len(ALL_STATIONS)))
    ax.set_xticklabels(ALL_STATIONS)
    ax.set_yticklabels(ALL_STATIONS)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    for xtl, ytl, color in zip(ax.get_xticklabels(), ax.get_yticklabels(), COLOR_CYCLE):
        xtl.set_color(color)
        ytl.set_color(color)
        xtl.set_weight('bold')
        ytl.set_weight('bold')
    ax.set_xlabel('Train station', labelpad=10)
    ax.set_ylabel('Test station', labelpad=7)

    # Colorbar handling
    im_ax = None
    cax = None
    orientation = None
    if isinstance(colorbar, Axes):
        cax = colorbar
        orientation = 'horizontal'
    elif colorbar is True:
        im_ax = ax
    else:  # If colorbar is not an Axes instance or True, then no colorbar!
        cax = False
    if cax is not False:
        fig.colorbar(
            im,
            ax=im_ax,
            cax=cax,
            orientation=orientation,
            label='Accuracy score',
            ticks=plt.MultipleLocator(0.25),  # So 50% is shown!
            format=PercentFormatter(xmax=1),
        )

    # Add text
    for i in range(len(ALL_STATIONS)):
        for j in range(len(ALL_STATIONS)):
            this_score = scores[i, j]
            # Choose the best text color for contrast
            if this_score > 0.5:
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
                alpha=0.7,
            )

    # Show or print stats
    if diagonal_metrics:
        mean = scores.diagonal().mean()
        std = scores.diagonal().std()
        title = (
            f'$\mu_\mathrm{{diag}}$ = {mean:.0%}\n$\sigma_\mathrm{{diag}}$ = {std:.1%}'
        )
    else:
        mean = scores.mean()
        std = scores.std()
        title = f'$\mu$ = {mean:.0%}\n$\sigma$ = {std:.1%}'
    if show_stats:
        ax.set_title(title, loc='left')
        ax.set_title('Testing\n{}'.format(day.strftime('%-d %B')), loc='right')
    else:
        print('Testing {}'.format(day.strftime('%-d %B')))
        subscript = '_diag' if diagonal_metrics else ''
        print(f'mean{subscript} = {mean:.0%}')
        print(f'std{subscript} = {std:.1%}')

import tempfile
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np

from svm import ALL_DAYS, ALL_STATIONS
from svm.plotting import plot_generalization_matrix

FONT_SIZE = 14  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load scores to plot for panel (a)
MANUAL_SCORE_FILE = 'manual_filter_roll.npy'
manual_scores = np.load(
    WORKING_DIR / 'plot_scripts' / 'generalization' / MANUAL_SCORE_FILE
)

# Load scores to plot for panel (b)
TSFRESH_SCORE_FILE = 'tsfresh_filter_roll.npy'
tsfresh_scores = np.load(
    WORKING_DIR / 'plot_scripts' / 'generalization' / TSFRESH_SCORE_FILE
)

#%% Define functions

# Box params
TRAIN_COLOR = '#edc948'
TEST_COLOR = '#e15759'
LW = 2


# Box origin is top-left of plot
def _box(xmin, xmax, ymin, ymax):
    return np.array([[xmin, xmax, xmax, xmin, xmin], [ymax, ymax, ymin, ymin, ymax]])


# Another helper function
def _find_train_min_max(test_ind, axis_length):
    all_values = np.arange(-0.5, axis_length, 0.5)
    not_in = np.setdiff1d(all_values, [test_ind])
    arr_split = np.split(not_in, [np.argwhere(np.diff(not_in) != 0.5).squeeze() + 1])
    return [(arr.min(), arr.max()) for arr in arr_split if arr.size > 1]


# Function for plotting the overlays
def train_test_overlay(ax, test_x_ind, test_y_ind):
    line_kwargs = dict(
        linewidth=LW,
        zorder=3,
        clip_on=False,
        solid_joinstyle='miter',
        dash_joinstyle='miter',
    )
    # Train box(es)
    for x_min_max in _find_train_min_max(test_x_ind, axis_length=len(ALL_DAYS)):
        for y_min_max in _find_train_min_max(test_y_ind, axis_length=len(ALL_STATIONS)):
            ax.plot(*_box(*x_min_max, *y_min_max), color=TRAIN_COLOR, **line_kwargs)
    # Test box
    ax.plot(
        *_box(
            test_x_ind - 0.5,
            test_x_ind + 0.5,
            test_y_ind - 0.5,
            test_y_ind + 0.5,
        ),
        color=TEST_COLOR,
        **line_kwargs,
    )


#%% Save images

temp_dir = tempfile.TemporaryDirectory()

for x_ind in range(len(ALL_DAYS)):
    for y_ind in range(len(ALL_STATIONS)):

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(
            ncols=3, width_ratios=[3, 3, 0.1]
        )  # Ratios set colorbar width
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        cax = fig.add_subplot(gs[2])

        # Call function
        plot_generalization_matrix(
            manual_scores, fig, ax1, show_stats=False, colorbar=False
        )
        plot_generalization_matrix(
            tsfresh_scores, fig, ax2, show_stats=False, colorbar=cax
        )

        # Tweak
        ax2.set_ylabel('')
        ax2.tick_params(which='both', axis='y', labelleft=False)
        fig.subplots_adjust(wspace=0.15)

        # Adjust colorbar position and height
        pos1 = ax1.get_position()
        posc = cax.get_position()
        cax.set_position([posc.x0, pos1.y0, posc.width, pos1.height])

        # Plot overlays
        for ax in ax1, ax2:
            train_test_overlay(ax, x_ind, y_ind)

        # Create legend using dummy entries
        for label, color in zip(['Train', 'Test'], [TRAIN_COLOR, TEST_COLOR]):
            cax.axvspan(
                np.nan, np.nan, facecolor='none', edgecolor=color, lw=LW, label=label
            )
        cax.legend(
            frameon=False,
            bbox_to_anchor=(-1.1, 1.06),
            loc='lower left',
            borderaxespad=0,
        )

        # Plot (a) and (b) tags
        for ax, label in zip([ax1, ax2], ['A', 'B']):
            ax.text(
                -0.04,
                1.03,
                label,
                ha='right',
                va='bottom',
                transform=ax.transAxes,
                weight='bold',
                fontsize=18,
            )

        fig.savefig(
            Path(temp_dir.name) / f'{x_ind}_{y_ind}.png', bbox_inches='tight', dpi=300
        )
        plt.close(fig)

#%% Read images and save GIF

imageio.mimwrite(
    Path.home() / 'Downloads' / 'generalization.gif',
    [imageio.imread(file) for file in sorted(Path(temp_dir.name).glob('*.png'))],
    duration=1,  # [s]
    subrectangles=True,
)

temp_dir.cleanup()

import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from svm.plotting import plot_generalization_matrix

FONT_SIZE = 14  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Define project directory
WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()

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

#%% Plot

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(ncols=3, width_ratios=[3, 3, 0.1])  # Ratios set colorbar width
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
cax = fig.add_subplot(gs[2])

# Call function
plot_generalization_matrix(manual_scores, fig, ax1, show_stats=False, colorbar=False)
plot_generalization_matrix(tsfresh_scores, fig, ax2, show_stats=False, colorbar=cax)

# Tweak
ax2.set_ylabel('')
ax2.tick_params(which='both', axis='y', labelleft=False)
fig.subplots_adjust(wspace=0.15)

# Adjust colorbar position and height
pos1 = ax1.get_position()
posc = cax.get_position()
cax.set_position([posc.x0, pos1.y0, posc.width, pos1.height])

# Plot (a) and (b) tags
for ax, label in zip([ax1, ax2], ['(a)', '(b)']):
    ax.text(
        -0.02,
        1.03,
        label,
        ha='right',
        va='bottom',
        transform=ax.transAxes,
        weight='bold',
        fontsize=18,
    )

fig.show()

_ = subprocess.run(['open', os.environ['YASUR_FIGURE_DIR']])

# fig.savefig(Path(os.environ['YASUR_FIGURE_DIR']).expanduser().resolve() / 'generalization.png', bbox_inches='tight', dpi=300)

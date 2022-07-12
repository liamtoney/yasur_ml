import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime

from svm.plotting import plot_path_effect_matrix

FONT_SIZE = 14  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Define project directory
WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()

# Load scores to plot (these scores created using `tsfresh_filter_roll.feather`)
SCORE_FILE = '2016-08-01.npy'
scores = np.load(WORKING_DIR / 'plot_scripts' / 'time_shift_tsfresh' / SCORE_FILE)

# Plot (gridspec stuff retained here to guarantee an identical plot to Fig. 4a)
fig = plt.figure(figsize=(15, 15))

gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[3, 0.1, 3])
ax1 = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[1, 0])

plot_path_effect_matrix(
    scores,
    fig,
    ax1,
    day=UTCDateTime(SCORE_FILE.rstrip('.npy')),
    colorbar=cax,
    show_stats=False,
    diagonal_metrics=True,
)

# Adjust cax
pos1 = ax1.get_position()
cbar = cax.get_position()
cax.set_position([pos1.x0, pos1.y0 - (3 * cbar.height), pos1.width, cbar.height])

fig.show()

_ = subprocess.run(['open', os.environ['YASUR_FIGURE_DIR']])

# fig.savefig(Path(os.environ['YASUR_FIGURE_DIR']).expanduser().resolve() / 'time_shift_tsfresh.pdf', bbox_inches='tight', dpi=800)

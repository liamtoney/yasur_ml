from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

#%% Plot

for scores in manual_scores, tsfresh_scores:
    fig, ax = plt.subplots()
    plot_generalization_matrix(scores, fig, ax)
    fig.tight_layout()
    fig.show()

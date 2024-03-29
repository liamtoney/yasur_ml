import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from svm.tools import balance_classes, read_and_preprocess

# Define project directory
WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()

# Filename of features to use
FEATURES_FILE = 'manual_filter.feather'

# Read in labeled features
features = read_and_preprocess(WORKING_DIR / 'features' / 'feather' / FEATURES_FILE)

# Balance classes for fair plotting comparison
features = balance_classes(features)

#%% Plot two features against each other as a scatter plot

X_AXIS_FEATURE = 'td_skewness'
Y_AXIS_FEATURE = 'td_kurtosis'

colors = [os.environ[f'SUBCRATER_{label}'] for label in features.label]

fig, ax = plt.subplots()
ax.scatter(
    features[X_AXIS_FEATURE],
    features[Y_AXIS_FEATURE],
    edgecolors=colors,
    facecolors='none',
)
ax.set_xlabel(X_AXIS_FEATURE)
ax.set_ylabel(Y_AXIS_FEATURE)
ax.set_title(f'{features.shape[0]} waveforms')

# Add legend
ax.scatter(
    [], [], edgecolors=os.environ['SUBCRATER_S'], facecolors='none', label='Subcrater S'
)
ax.scatter(
    [], [], edgecolors=os.environ['SUBCRATER_N'], facecolors='none', label='Subcrater N'
)
ax.legend()

fig.show()

#%% Histograms of each feature

feature_names = features.columns[3:]  # Skip first three columns

MANUAL_BIN_RANGE = True

ALPHA = 0.7  # Transparency for histograms (to see overlap)

NCOLS = 3  # Number of subplot columns
NBINS = 50  # Number of histogram bins

fig, axes = plt.subplots(
    nrows=int(np.ceil(len(feature_names) / NCOLS)), ncols=NCOLS, figsize=(8, 10)
)

for ax, feature in zip(axes.flatten(), feature_names):
    bin_range = None
    if MANUAL_BIN_RANGE:
        if feature == 'fd_peak':
            bin_range = (0, 6)
        elif feature == 'fd_q1':
            bin_range = (0, 5)
        elif feature == 'fd_q2':
            bin_range = (0, 8)
        elif feature == 'fd_q3':
            bin_range = (0, 12)
        elif feature == 'td_kurtosis':
            bin_range = (-5, 40)
        elif feature == 'td_skewness':
            bin_range = (-2, 5)
    ax.hist(
        features[features.label == 'S'][feature],
        bins=NBINS,
        range=bin_range,
        color=os.environ['SUBCRATER_S'],
        label='Subcrater S',
        alpha=ALPHA,
    )
    ax.hist(
        features[features.label == 'N'][feature],
        range=bin_range,
        bins=NBINS,
        color=os.environ['SUBCRATER_N'],
        label='Subcrater N',
        alpha=ALPHA,
    )
    ax.set_title(feature)

# Remove empty subplots, if any
for ax in axes.flatten():
    if not ax.has_data():
        fig.delaxes(ax)

fig.suptitle(f'{features.shape[0]} waveforms')
fig.tight_layout()

# Add legend
last_ax = axes.flatten()[len(feature_names) - 1]
last_ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left', frameon=False)

fig.show()

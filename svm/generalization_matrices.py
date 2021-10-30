import json
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from obspy import UTCDateTime
from sklearn import preprocessing, svm

from svm.tools import balance_classes, format_scikit, read_and_preprocess, time_subset

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Read in features only once, since it's slow
features_all = read_and_preprocess(
    WORKING_DIR / 'features' / 'csv' / 'manual_filter_roll.csv'
)

#%% Subset features (only applies for TSFRESH features)

is_tsfresh = 'tsfresh' in features_all.attrs['filename']

if is_tsfresh:
    with open(
        WORKING_DIR
        / 'features'
        / 'selected_names'
        / 'SFS_features_tsfresh_roll_filtered.json'
    ) as f:
        top_features = json.load(f)
    features = features_all[features_all.columns[:3].tolist() + top_features]
else:
    features = features_all

#%% Run function

# Define number of runs [= random calls to balance_classes()] to perform and average
RUNS = 10

ALL_STATIONS = [f'YIF{n}' for n in range(1, 6)]

ALL_DAYS = [
    UTCDateTime(2016, 7, 27),
    UTCDateTime(2016, 7, 28),
    UTCDateTime(2016, 7, 29),
    UTCDateTime(2016, 7, 30),
    UTCDateTime(2016, 7, 31),
    UTCDateTime(2016, 8, 1),
]

# Preallocate scores matrix
scores = np.empty((len(ALL_STATIONS), len(ALL_DAYS)))

# Iterate over days
for j, tmin in enumerate(ALL_DAYS):

    print(tmin.strftime('%-d %B'))

    # 24 hrs
    tmax = tmin + 24 * 60 * 60

    # Temporal subsetting
    train, test, *_ = time_subset(features, 'test', tmin, tmax)

    # Preallocate scores for this day
    station_scores = np.empty((len(ALL_STATIONS), RUNS))

    # Perform this loop RUNS times
    for k in range(RUNS):

        if RUNS > 1:
            print(f'\t{k + 1}/{RUNS}')

        # Balance classes (RANDOM!)
        train_bal = balance_classes(train, verbose=False)
        test_bal = balance_classes(test, verbose=False)

        # Iterate over test stations
        for i, test_station in enumerate(ALL_STATIONS):

            # Training subset
            train_ds = train_bal[train_bal.station != test_station]  # Subset
            X_train, y_train = format_scikit(train_ds)
            X_train = preprocessing.scale(X_train)  # Rescale

            # Testing subset
            test_ds = test_bal[test_bal.station == test_station]  # Subset
            X_test, y_test = format_scikit(test_ds)
            X_test = preprocessing.scale(X_test)  # Rescale

            # Fit SVC
            clf = svm.LinearSVC(dual=False)
            clf.fit(X_train, y_train)

            # Test SVC
            station_scores[i, k] = clf.score(X_test, y_test)

    scores[:, j] = station_scores.mean(axis=1)  # Take mean of the RUNS runs

# Make plot
fig, ax = plt.subplots()
im = ax.imshow(scores, cmap=cc.m_diverging_bwr_20_95_c54_r, vmin=0, vmax=1)
ax.set_xticks(range(len(ALL_DAYS)))
ax.set_yticks(range(len(ALL_STATIONS)))
ax.set_xticklabels([d.strftime('%-d\n%B') for d in ALL_DAYS])
ax.set_yticklabels(ALL_STATIONS)
ax.set_xlabel('Test day', weight='bold', labelpad=10)
ax.set_ylabel('Test station', weight='bold', labelpad=5)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Colorbar
fig.colorbar(
    im,
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

# Add title
ax.set_title(f'$\mu$ = {scores.mean():.0%}\n$\sigma$ = {scores.std():.1%}', loc='left')

fig.tight_layout()
fig.show()

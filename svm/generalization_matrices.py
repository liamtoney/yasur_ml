import json
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from sklearn import preprocessing, svm

from svm.plotting import plot_generalization_matrix
from svm.tools import balance_classes, format_scikit, read_and_preprocess, time_subset

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Read in features only once, since it's slow
features_all = read_and_preprocess(
    WORKING_DIR / 'features' / 'feather' / 'tsfresh_filter_roll.feather'
)

ALL_STATIONS = [f'YIF{n}' for n in range(1, 6)]

ALL_DAYS = [
    UTCDateTime(2016, 7, 27),
    UTCDateTime(2016, 7, 28),
    UTCDateTime(2016, 7, 29),
    UTCDateTime(2016, 7, 30),
    UTCDateTime(2016, 7, 31),
    UTCDateTime(2016, 8, 1),
]

#%% Subset features (only applies for TSFRESH features)

is_tsfresh = 'tsfresh' in features_all.attrs['filename']

if is_tsfresh:
    with open(
        WORKING_DIR
        / 'features'
        / 'selected_names'
        / f'SFS_{features_all.attrs["filename"]}_10_r00.json'
    ) as f:
        top_features = json.load(f)
    features = features_all[features_all.columns[:3].tolist() + top_features]
else:
    features = features_all

#%% Run function

# Define number of runs [= random calls to balance_classes()] to perform and average
RUNS = 10

# Toggle exporting 5 x 6 scores matrices for plotting outside this script
EXPORT_SCORES = True

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

if EXPORT_SCORES:
    features_file = features.attrs['filename']
    np.save(
        WORKING_DIR / 'plot_scripts' / 'generalization' / f'{features_file}.npy', scores
    )

# Make plot
plot_generalization_matrix(scores)

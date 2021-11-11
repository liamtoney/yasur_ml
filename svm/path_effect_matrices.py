import json
from pathlib import Path

import numpy as np
from sklearn import preprocessing, svm

from svm import ALL_DAYS, ALL_STATIONS
from svm.plotting import plot_path_effect_matrix
from svm.tools import balance_classes, format_scikit, read_and_preprocess, time_subset

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Read in features only once, since it's slow
features_all = read_and_preprocess(
    WORKING_DIR / 'features' / 'feather' / 'tsfresh_filter.feather'
)

#%% Subset features (OPTIONAL; only applies for TSFRESH features)

SUBSET = False

if SUBSET:
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
RUNS = 5

# Toggle plotting mean / std for ALL entries or just the diagonal
DIAGONAL_METRICS = True

# Toggle exporting 5 x 5 scores matrices for plotting outside this script
EXPORT_SCORES = False

for tmin in ALL_DAYS:

    print(tmin.strftime('%-d %B'))

    # 24 hrs
    tmax = tmin + 24 * 60 * 60

    # Temporal subsetting
    train, test, *_ = time_subset(features, 'test', tmin, tmax)

    # Preallocate scores for this day
    run_scores = np.empty((len(ALL_STATIONS), len(ALL_STATIONS), RUNS))

    # Perform this loop RUNS times
    for k in range(RUNS):

        if RUNS > 1:
            print(f'\t{k + 1}/{RUNS}')

        # Balance classes (RANDOM!)
        train_bal = balance_classes(train, verbose=False)
        test_bal = balance_classes(test, verbose=False)

        for j, train_station in enumerate(ALL_STATIONS):

            train_ds = train_bal[train_bal.station == train_station]  # Subset
            X_train, y_train = format_scikit(train_ds)
            X_train = preprocessing.scale(X_train)  # Rescale

            # Fit SVC
            clf = svm.LinearSVC(dual=False)
            clf.fit(X_train, y_train)

            for i, test_station in enumerate(ALL_STATIONS):

                test_ds = test_bal[test_bal.station == test_station]  # Subset
                X_test, y_test = format_scikit(test_ds)
                X_test = preprocessing.scale(X_test)  # Rescale

                # Test SVC
                run_scores[i, j, k] = clf.score(X_test, y_test)

    scores = run_scores.mean(axis=2)  # Take mean of the RUNS runs

    if EXPORT_SCORES:
        day_str = tmin.strftime('%Y-%m-%d')
        np.save(
            WORKING_DIR / 'plot_scripts' / 'path_effects' / f'{day_str}.npy', scores
        )

    # Make plot
    plot_path_effect_matrix(scores, day=tmin, diagonal_metrics=DIAGONAL_METRICS)

import json
from pathlib import Path

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
    WORKING_DIR / 'features' / 'feather' / 'tsfresh_filter.feather'
)

# Define new color cycle based on entries 3–7 in "New Tableau 10", see
# https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-56782
COLOR_CYCLE = ['#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']

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

ALL_STATIONS = [f'YIF{n}' for n in range(1, 6)]

for tmin in [
    UTCDateTime(2016, 7, 27),
    UTCDateTime(2016, 7, 28),
    UTCDateTime(2016, 7, 29),
    UTCDateTime(2016, 7, 30),
    UTCDateTime(2016, 7, 31),
    UTCDateTime(2016, 8, 1),
]:

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

    # Make plot
    fig, ax = plt.subplots()
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
    ax.set_xlabel('Train station', weight='bold', labelpad=10)
    ax.set_ylabel('Test station', weight='bold', labelpad=7)

    # Colorbar
    fig.colorbar(
        im,
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

    # Add titles
    if DIAGONAL_METRICS:
        title = f'$\mu_\mathrm{{diag}}$ = {scores.diagonal().mean():.0%}\n$\sigma_\mathrm{{diag}}$ = {scores.diagonal().std():.1%}'
    else:
        title = f'$\mu$ = {scores.mean():.0%}\n$\sigma$ = {scores.std():.1%}'
    ax.set_title(title, loc='left')
    ax.set_title('Testing\n{}'.format(tmin.strftime('%-d %B')), loc='right')

    fig.tight_layout()
    fig.show()

    # fig.savefig(
    #     f'/Users/ldtoney/Downloads/path_effect_scores_{tmin.month}-{tmin.day}.png',
    #     bbox_inches='tight',
    #     dpi=300,
    # )

import sys
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from sklearn import preprocessing, svm

# TODO: Must add svm/ to path to import, worth making into a package?
sys.path.append('svm/')
from train_test import (balance_classes, format_scikit, read_and_preprocess,
                        time_subset)

WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Read in features only once, since it's slow
features = read_and_preprocess(
    WORKING_DIR / 'features' / 'csv' / 'features_tsfresh.csv'
)

#%% Run function

ALL_STATIONS = [f'YIF{n}' for n in range(1, 6)]

TIME_WINDOW_TYPE = 'test'

for tmin in [
    UTCDateTime(2016, 7, 27),
    UTCDateTime(2016, 7, 28),
    UTCDateTime(2016, 7, 29),
    UTCDateTime(2016, 7, 30),
    UTCDateTime(2016, 7, 31),
    UTCDateTime(2016, 8, 1),
]:

    # 24 hrs
    tmax = tmin + 24 * 60 * 60

    # Preallocate scores matrix
    scores = np.empty((len(ALL_STATIONS), len(ALL_STATIONS)))

    # Temporal subsetting
    train, test, *_ = time_subset(features, TIME_WINDOW_TYPE, tmin, tmax)

    # Balance classes
    print('TRAINING')
    train = balance_classes(train)
    print('\nTESTING')
    test = balance_classes(test)

    for j, train_station in enumerate(ALL_STATIONS):

        train_ds = train[train.station == train_station]  # Subset
        X_train, y_train = format_scikit(train_ds)
        X_train = preprocessing.scale(X_train)  # Rescale

        # Fit SVC
        clf = svm.LinearSVC(dual=False)
        clf.fit(X_train, y_train)

        for i, test_station in enumerate(ALL_STATIONS):

            test_ds = test[test.station == test_station]  # Subset
            X_test, y_test = format_scikit(test_ds)
            X_test = preprocessing.scale(X_test)  # Rescale

            # Test SVC
            scores[i, j] = clf.score(X_test, y_test)

    # Make plot
    fig, ax = plt.subplots()
    im = ax.imshow(scores, cmap=cc.m_diverging_bwr_20_95_c54_r, vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_STATIONS)))
    ax.set_yticks(range(len(ALL_STATIONS)))
    ax.set_xticklabels(ALL_STATIONS)
    ax.set_yticklabels(ALL_STATIONS)
    ax.set_xlabel('TRAIN', weight='bold', labelpad=10)
    ax.set_ylabel('TEST', weight='bold', labelpad=5)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Handle time limits
    if tmin < features.time.min():
        tmin = features.time.min()
    if tmax > features.time.max():
        tmax = features.time.max()
    fmt = '%Y-%m-%d %H:%M'
    ax.set_title(
        f'{TIME_WINDOW_TYPE.capitalize()}ing window: {tmin.strftime(fmt)} – {tmax.strftime(fmt)}',
        pad=20,
    )

    # Colorbar formatting
    cbar = fig.colorbar(im, label='Score (%)')
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{t * 100:.0f}' for t in ticks])

    # Add text
    for i in range(len(ALL_STATIONS)):
        for j in range(len(ALL_STATIONS)):
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

    fig.tight_layout()
    fig.show()

    # fig.savefig(
    #     f'/Users/ldtoney/Downloads/path_effect_scores_{tmin.month}-{tmin.day}.png',
    #     bbox_inches='tight',
    #     dpi=300,
    # )

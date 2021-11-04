#!/usr/bin/env python

import json
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.feature_selection import RFE, SequentialFeatureSelector

from svm.tools import (
    balance_classes,
    format_scikit,
    read_and_preprocess,
    remove_correlated_features,
)

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Read in features only once, since it's slow
features_all = read_and_preprocess(
    WORKING_DIR / 'features' / 'feather' / 'tsfresh_filter_roll.feather'
)

#%% Remove correlated features (vastly speeds up subsequent feature selection!)

PRUNE_FEATURES = False

if PRUNE_FEATURES:
    features = remove_correlated_features(features_all, thresh=1)
else:
    features = features_all

#%% Pre-process and define function

# Define # of features to select
N_FEATURES = 10

# Balance classes
ds = balance_classes(features, verbose=True, random_state=None)

# Format dataset for use with scikit-learn
X, y = format_scikit(ds)

# Rescale data to have zero mean and unit variance
X = preprocessing.scale(X)


# Define function to plot top features and export to JSON
def plot_and_export(selector, prefix):

    top_feature_names = features.columns[3:][selector.get_support()]

    feature_values = selector.transform(X)

    fig, axes = plt.subplots(nrows=top_feature_names.size, sharex=True)
    for ax, values, feature_name in zip(axes, feature_values.T, top_feature_names):
        ax.plot(values)
        ax.set_title(feature_name)
    fig.show()

    # Save, incrementing filename to avoid overwriting previous runs
    template = str(
        WORKING_DIR
        / 'features'
        / 'selected_names'
        / f'{prefix}_{features.attrs["filename"]}_{N_FEATURES}_r{{:02}}.json'
    )
    run_no = 0
    while Path(template.format(run_no)).exists():
        run_no += 1
    with open(template.format(run_no), 'w') as f:
        json.dump(top_feature_names.tolist(), f, indent=2)


#%% Run RFE

print('Running RFE')
t1 = time()
rfe = RFE(
    estimator=svm.LinearSVC(dual=False),
    n_features_to_select=N_FEATURES,
    step=0.1,
    verbose=1,
)
rfe.fit(X, y)

#%% Plot and export RFE results

plot_and_export(rfe, prefix='RFE')
t2 = time()
print(f'Done (elapsed time: {(t2 - t1) / 60:.0f} min)')

#%% Run SFS

print('Running SFS')
t1 = time()
sfs = SequentialFeatureSelector(
    estimator=svm.LinearSVC(dual=False),
    n_features_to_select=N_FEATURES,
    direction='forward',
    scoring='accuracy',
    cv=5,  # Could increase this but would be slower
    n_jobs=-1,
)
sfs.fit(X, y)

#%% Plot and export SFS results

plot_and_export(sfs, prefix='SFS')
t2 = time()
print(f'Done (elapsed time: {(t2 - t1) / 60:.0f} min)')

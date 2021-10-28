import json
from pathlib import Path

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
    WORKING_DIR / 'features' / 'csv' / 'features_tsfresh_roll_filtered.csv'
)

#%% Remove correlated features (vastly speeds up subsequent feature selection!)

features = remove_correlated_features(features_all, thresh=None)

#%% Pre-process and define function

# Define # of features to select
N_FEATURES = 10

# Balance classes
ds = balance_classes(features, verbose=True)

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

    fname = prefix + '_' + features.attrs['filename'] + '.json'
    with open(WORKING_DIR / 'features' / 'selected_names' / fname, 'w') as f:
        json.dump(top_feature_names.tolist(), f, indent=2)


#%% Run RFE

rfe = RFE(estimator=svm.LinearSVC(dual=False), n_features_to_select=N_FEATURES, step=1)
rfe.fit(X, y)

#%% Plot and export RFE results

plot_and_export(rfe, prefix='RFE')

#%% Run SFS

sfs = SequentialFeatureSelector(
    estimator=svm.LinearSVC(dual=False),
    n_features_to_select=N_FEATURES,
    direction='forward',
    scoring='accuracy',
    cv=2,  # Could increase this but would be slower
    n_jobs=-1,
)
sfs.fit(X, y)

#%% Plot and export SFS results

plot_and_export(sfs, prefix='SFS')

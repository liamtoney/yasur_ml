import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from train_test import balance_classes, format_scikit, read_and_preprocess

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

#%% Just load features once

FEATURES_CSV = 'features_tsfresh_roll_filtered.csv'
features = read_and_preprocess(WORKING_DIR / 'features' / 'csv' / FEATURES_CSV)

#%% Pre-process and define function

# Balance classes
ds = balance_classes(features, verbose=True)

# Format dataset for use with scikit-learn
X, y = format_scikit(ds)

# Rescale data to have zero mean and unit variance
X = preprocessing.scale(X)


# Define function to plot top features and export to JSON
def plot_and_export(selector):

    top_feature_names = features.columns[3:][selector.get_support()]

    feature_values = selector.transform(X)

    fig, axes = plt.subplots(nrows=top_feature_names.size, sharex=True)
    for ax, values, feature_name in zip(axes, feature_values.T, top_feature_names):
        ax.plot(values)
        ax.set_title(feature_name)
    fig.show()

    fname = type(selector).__name__ + '_' + FEATURES_CSV.replace('.csv', '.json')
    with open(WORKING_DIR / 'features' / 'selected_names' / fname, 'w') as f:
        json.dump(top_feature_names.tolist(), f, indent=2)


#%% Run RFE

rfe = RFE(estimator=svm.LinearSVC(dual=False), n_features_to_select=10, step=50)
rfe.fit(X, y)  # Takes a LONG time, could adjust step above to help

#%% Plot and export RFE results

plot_and_export(rfe)

#%% Run SFS

sfs = SequentialFeatureSelector(
    estimator=svm.LinearSVC(dual=False),
    n_features_to_select=10,
    direction='forward',
    scoring='accuracy',
    cv=2,  # Could increase this but would be slower
    n_jobs=-1,
)
sfs.fit(X, y)  # Takes a while but not nearly as long as RFE

#%% Plot and export SFS results

plot_and_export(sfs)

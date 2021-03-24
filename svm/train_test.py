#!/usr/bin/env python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from sklearn import preprocessing, svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Toggle plotting
PLOT = False

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Filename of features CSV to use
FEATURES_CSV = 'features_tsfresh.csv'

# Set to integer for reproducible results
RANDOM_STATE = None

# Fraction of data to use for training
TRAIN_SIZE = 0.75

# Read in labeled features
features = pd.read_csv(WORKING_DIR / 'features' / 'csv' / FEATURES_CSV)

# Convert times to UTCDateTime
features.time = [UTCDateTime(t) for t in features.time]

# Remove constant features
for column in features.columns:
    if np.unique(features[column]).size == 1:
        features.drop(columns=[column], inplace=True)

# Remove rows with NaNs
features.dropna(inplace=True)

# Adjust for class imbalance by down-sampling the majority class (from
# https://elitedatascience.com/imbalanced-classes)
class_counts_before = features.label.value_counts()
print('Before:\n' + class_counts_before.to_string())
dominant_vent = class_counts_before.index[class_counts_before.argmax()]
majority = features[features.label == dominant_vent]
minority = features[features.label != dominant_vent]
majority_downsampled = resample(
    majority,
    replace=False,  # Sample w/o replacement
    n_samples=minority.shape[0],  # Match number of waveforms in minority class
    random_state=RANDOM_STATE,
)
features_downsampled = pd.concat([majority_downsampled, minority])
class_counts_after = features_downsampled.label.value_counts()
print('After:\n' + class_counts_after.to_string())
num_removed = (class_counts_before - class_counts_after)[dominant_vent]
print(f'({num_removed} vent {dominant_vent} examples removed)')

# Format dataset for use with scikit-learn
y = (features_downsampled['label'] == 'C').to_numpy(dtype=int)  # 0 = vent A; 1 = vent C
X = features_downsampled.iloc[:, 2:].to_numpy()  # Skipping first two columns here

# Rescale data to have zero mean and unit variance
X_scaled = preprocessing.scale(X)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
)
print(f'\nTraining portion: {TRAIN_SIZE * 100:g}%')
print(f'Training size: {y_train.size}')
print(f'Testing size: {y_test.size}')

# Run SVC
clf = svm.LinearSVC(max_iter=5000)
clf.fit(X_train, y_train)

# Test SVC
score = clf.score(X_test, y_test)
print(f'\nAccuracy is {score:.0%}')

# Plot
cm = plot_confusion_matrix(
    clf,
    X_test,
    y_test,
    labels=[0, 1],  # Explicitly setting the order here
    display_labels=['A', 'C'],  # Since 0 = vent A; 1 = vent C
    cmap='Greys',
    normalize='true',  # 'true' means the diagonal contains the TPR and TNR
    values_format='.0%',  # Format as integer percent
)
fig = cm.figure_
ax = fig.axes[0]
ax.set_xlabel(ax.get_xlabel().replace('label', 'vent'))
ax.set_ylabel(ax.get_ylabel().replace('label', 'vent'))
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_weight('bold')
    label.set_color(os.environ[f'VENT_{label.get_text()}'])
ax.set_title(f'{y_test.size:,} test waveforms')
fig.axes[1].remove()  # Remove colorbar
fig.tight_layout()
fig.show()

#%% (DRAFT) Feature importance plot (permutation) - bad for correlated features

if PLOT:

    import numpy as np
    from sklearn.inspection import permutation_importance

    # Use training data for this
    result = permutation_importance(clf, X_train, y_train, n_repeats=10)
    sorted_idx = result.importances_mean.argsort()

    feature_names = features.columns[2:]
    y_ticks = np.arange(0, len(feature_names))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, result.importances_mean[sorted_idx], color='grey')
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_xlim(left=0)
    ax.set_xlabel('Relative feature importance')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.show()

    #%% (DRAFT) Feature importance plot (RF)

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier

    forest = ExtraTreesClassifier(n_estimators=250, random_state=47)

    # Use training data for this
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    sorted_idx = np.argsort(importances)

    feature_names = features.columns[2:]
    y_ticks = np.arange(0, len(feature_names))
    fig, ax = plt.subplots()
    ax.barh(
        y_ticks,
        importances[sorted_idx],
        color='grey',
        xerr=std[sorted_idx],
    )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(feature_names[sorted_idx])

    ax.set_xlabel('Relative feature importance')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fig.tight_layout()
    fig.show()

    #%% (DRAFT) Look at feature correlation

    import numpy as np
    from scipy.cluster import hierarchy
    from scipy.stats import spearmanr

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    feature_names = features.columns[2:]
    corr = spearmanr(X).correlation
    corr_linkage = hierarchy.ward(corr)
    dendro = hierarchy.dendrogram(
        corr_linkage, labels=feature_names, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))
    ax1.set_yticks([])

    # Take absolute value since we don't want anticorrelated features either
    ax2.imshow(
        np.abs(corr[dendro['leaves'], :][:, dendro['leaves']]),
        cmap='Greys',
        vmin=0,
        vmax=1,
    )
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    ax2.set_title('Absolute value of correlation')

    fig.tight_layout()
    fig.show()

    #%% (DRAFT) Plot feature correlation via scatterplot, plot decision boundary if possible

    import numpy as np

    ind1 = 0
    ind2 = 1

    # Use testing data
    f1 = X_test[:, ind1]
    f2 = X_test[:, ind2]
    colors = [os.environ['VENT_A'] if l == 0 else os.environ['VENT_C'] for l in y_test]

    fig, ax = plt.subplots()
    ax.scatter(f1, f2, edgecolors=colors, facecolors='none')
    ax.set_xlabel(feature_names[ind1])
    ax.set_ylabel(feature_names[ind2])

    # Add legend
    ax.scatter(
        [], [], edgecolors=os.environ['VENT_A'], facecolors='none', label='Vent A'
    )
    ax.scatter(
        [], [], edgecolors=os.environ['VENT_C'], facecolors='none', label='Vent C'
    )
    ax.legend()

    # If 2D feature space, plot decision line
    if X_scaled.shape[1] == 2:
        y = (
            lambda x: -(clf.coef_[0][0] / clf.coef_[0][1]) * x
            - clf.intercept_[0] / clf.coef_[0][1]
        )
        x = np.linspace(f1.min(), f1.max(), 100)
        ax.plot(x, y(x), color='black', label='Decision')
        ax.set_xlim(f1.min(), f1.max())
        ax.set_ylim(f2.min(), f2.max())

    fig.show()

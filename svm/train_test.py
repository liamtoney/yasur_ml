from pathlib import Path

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Filename of features CSV to use
FEATURES_CSV = 'features_filtered.csv'

# Set to integer for reproducible results
RANDOM_STATE = None

# Fraction of data to use for training
TRAIN_SIZE = 0.75

# Read in labeled features
features = pd.read_csv(WORKING_DIR / 'features' / FEATURES_CSV)

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
X = features_downsampled.iloc[:, 1:].to_numpy()

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
ax.set_title(f'{y_test.size:,} test waveforms')
fig.axes[1].remove()  # Remove colorbar
fig.tight_layout()
fig.show()

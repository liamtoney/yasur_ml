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

# Toggle plotting (for script use, mainly)
PLOT = True

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Maximum iterations for SVC classifier
MAX_ITER = 10000


def read_and_preprocess(features_csv_file):
    """Read in a features CSV file and perform basic pre-processing.

    Args:
        features_csv_file (str or Path): Full path to input features CSV file (must have
            a "time" column)

    Returns:
        pandas.DataFrame: Output features
    """

    # Read in labeled features
    features = pd.read_csv(features_csv_file)

    # Convert times to UTCDateTime
    features.time = [UTCDateTime(t) for t in features.time]

    # Remove constant features
    for column in features.columns:
        if np.unique(features[column]).size == 1:
            features.drop(columns=[column], inplace=True)

    # Remove rows with NaNs
    features.dropna(inplace=True)

    return features


def balance_classes(features):
    """Function to adjust for class imbalance by down-sampling the majority class.

    See https://elitedatascience.com/imbalanced-classes for more info.

    Args:
        features (pandas.DataFrame): Input features (must have a "label" column)

    Returns:
        pandas.DataFrame: Output features
    """

    # Perform the balancing
    class_counts_before = features.label.value_counts()
    print('Before:\n' + class_counts_before.to_string())
    dominant_vent = class_counts_before.index[class_counts_before.argmax()]
    majority = features[features.label == dominant_vent]
    minority = features[features.label != dominant_vent]
    majority_downsampled = resample(
        majority,
        replace=False,  # Sample w/o replacement
        n_samples=minority.shape[0],  # Match number of waveforms in minority class
        random_state=None,
    )
    features_downsampled = pd.concat([majority_downsampled, minority])

    # Print what action was taken
    class_counts_after = features_downsampled.label.value_counts()
    print('After:\n' + class_counts_after.to_string())
    num_removed = (class_counts_before - class_counts_after)[dominant_vent]
    print(f'({num_removed} vent {dominant_vent} examples removed)')

    return features_downsampled


def format_scikit(features):
    """Format a features DataFrame for use with scikit-learn.

    Args:
        features (pandas.DataFrame): Input features (must have "station", "time", and
            "label" as the first three columns)

    Returns:
        Tuple containing X and y (NumPy arrays)
    """

    X = features.iloc[:, 3:].to_numpy()  # Skipping first three (metadata) columns
    y = (features['label'] == 'C').to_numpy(dtype=int)  # 0 = vent A; 1 = vent C

    return X, y


def plot_confusion(clf, X_test, y_test, title=None):
    """Shallow wrapper around plot_confusion_matrix().

    See
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
    for more info. This function basically tweaks the standard plot for the Yasur
    context.

    Warning:
        Double-check that the colored "A" and "C" labels are set in boldface. The
        ~/.matplotlib/fontlist-v???.json font cache might need to be deleted and the
        code re-run to accomplish this.

    Args:
        clf (estimator): Fitted classifier
        X_test (numpy.ndarray): Input values
        y_test (numpy.ndarray): Target values
        title (str or None): Title for plot (no title if None)
    """

    cm = plot_confusion_matrix(
        clf,
        X_test,
        y_test,
        labels=[0, 1],  # Explicitly setting the order here
        display_labels=['A', 'C'],  # Since 0 = vent A; 1 = vent C
        cmap='Greys',
        normalize='true',  # 'true' means the diagonal contains the TPR and TNR
        values_format='.0%',  # Format as integer percent
        colorbar=False,  # Don't add colorbar
    )
    cm.im_.set_clim(0, 1)  # For easier comparison between plots
    fig = cm.figure_
    ax = fig.axes[0]
    ax.set_xlabel(ax.get_xlabel().replace('label', 'vent'))
    ax.set_ylabel(ax.get_ylabel().replace('label', 'vent'))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('bold')
        label.set_color(os.environ[f'VENT_{label.get_text()}'])
    if title is not None:
        ax.set_title(title, loc='left', fontsize='medium')
    fig.tight_layout()
    fig.show()


def time_subset(features, time_window_type, tmin, tmax):
    """Temporally split a features DataFrame.

    Args:
        features (pandas.DataFrame): Input features (must have "time" column)
        time_window_type (str): Either 'train' or 'test'
        tmin (UTCDateTime or None): Beginning of time window (None to use start of
            features)
        tmax (UTCDateTime or None): End of time window (None to use end of features)

    Returns:
        Tuple containing training subset, testing subset, tmin, and tmax
    """

    if tmin is None:
        tmin = features.time.min()
        print('Using beginning of features for tmin')
    if tmax is None:
        tmax = features.time.max()
        print('Using end of features for tmax')

    if time_window_type == 'train':
        train = features[(features.time >= tmin) & (features.time <= tmax)]
        test = features[(features.time < tmin) | (features.time > tmax)]
    elif time_window_type == 'test':
        train = features[(features.time < tmin) | (features.time > tmax)]
        test = features[(features.time >= tmin) & (features.time <= tmax)]
    else:
        raise ValueError('Window must be either \'train\' or \'test\'')

    if train.shape[0] + test.shape[0] != features.shape[0]:
        raise ValueError('Temporal subsetting failed due to dimension mismatch!')

    return train, test, tmin, tmax


def train_test(
    features_path,
    train_size=None,
    train_stations=[],
    test_stations=[],
    time_window_type=None,
    tmin=None,
    tmax=None,
    plot=False,
    random_state=None,
):
    """Train and test an SVM model using a features CSV file.

    Args:
        features_path (str): Full path to CSV file
        train_size (float or None): Random fraction of [balanced] data to use for
            training
        train_stations (str or list): Station(s) to train on
        test_stations (str or list): Station(s) to test on
        time_window_type (str or None): Either 'train', 'test', or None (the latter
            disables time subsetting)
        tmin (UTCDateTime or None): Beginning of time window (None to use start of
            features)
        tmax (UTCDateTime or None): End of time window (None to use end of features)
        plot (bool): Toggle plotting confusion matrix
        random_state (int or None): Set to integer for reproducible results
    """

    # Type conversion
    train_stations = np.atleast_1d(train_stations)
    test_stations = np.atleast_1d(test_stations)

    # Input checking (doing this before the potentially lengthy read-in step)
    if (
        train_size is None and (len(train_stations) == 0 or len(test_stations) == 0)
    ) or (
        train_size is not None and (len(train_stations) != 0 or len(test_stations) != 0)
    ):
        raise ValueError(
            'Either train_size OR train_stations AND test_stations must be set!'
        )

    # Read in labeled features (can be slow!)
    features = read_and_preprocess(features_path)

    if train_size is not None:  # Subset using random fraction of data

        # Balance classes
        features_downsampled = balance_classes(features)

        # Format dataset for use with scikit-learn
        X, y = format_scikit(features_downsampled)

        # Split into training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )

    else:  # Subset by time or by station

        # TIME subsetting
        if time_window_type is not None:
            train, test, tmin, tmax = time_subset(
                features, time_window_type, tmin, tmax
            )
        else:
            train, test = features, features

        # STATION subsetting
        train = train[train.station.isin(train_stations)]
        test = test[test.station.isin(test_stations)]

        # Balance train and test subsets
        print('TRAINING')
        train_ds = balance_classes(train)
        print('\nTESTING')
        test_ds = balance_classes(test)

        # Format dataset for use with scikit-learn
        X_train, y_train = format_scikit(train_ds)
        X_test, y_test = format_scikit(test_ds)

    print(f'\nTraining portion: {y_train.size / (y_train.size + y_test.size) * 100:g}%')
    print(f'Training size: {y_train.size}')
    print(f'Testing size: {y_test.size}')

    # Rescale data to have zero mean and unit variance
    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    # Run SVC
    clf = svm.LinearSVC(max_iter=MAX_ITER)
    clf.fit(X_train, y_train)

    # Test SVC
    score = clf.score(X_test, y_test)
    print(f'\nAccuracy is {score:.0%}')

    # Plot if desired
    if plot:
        title = 'Training stations: {}\nTesting stations: {}'.format(
            ', '.join(train_stations), ', '.join(test_stations)
        )
        fmt = '%Y-%m-%d %H:%M'
        if time_window_type is not None:
            title += f'\n{time_window_type.capitalize()}ing window: {tmin.strftime(fmt)} – {tmax.strftime(fmt)}'
        title += f'\n$\\bf{y_test.size}~test~waveforms$'
        plot_confusion(clf, X_test, y_test, title=title)

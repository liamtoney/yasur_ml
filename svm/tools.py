import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import plot_confusion_matrix
from sklearn.utils import resample
from tsfresh import select_features


def read_and_preprocess(features_feather_file):
    """Read in a features Feather file and perform basic pre-processing.

    Args:
        features_feather_file (str or Path): Full path to input features Feather file
            (must have a "time" column)

    Returns:
        pandas.DataFrame: Output features
    """

    # Read in labeled features
    features = pd.read_feather(features_feather_file)

    # Remove columns containing any number of NaNs
    features.dropna(axis='columns', how='any', inplace=True)

    # Remove constant features
    features = features.loc[:, features.nunique() != 1]

    # Convert times to UTCDateTime
    features.time = features.time.map(lambda t: UTCDateTime(t))

    # Reset index
    features.reset_index(drop=True, inplace=True)

    # Add filename metadata
    filename = Path(str(features_feather_file)).stem
    features.attrs['filename'] = filename

    return features


def tsfresh_select(features, **selection_kwargs):
    """Apply TSFRESH's select_features(), skipping metadata columns

    Args:
        features (pandas.DataFrame): Input features
        **selection_kwargs: Keyword arguments to be passed on to select_features()

    Returns:
        pandas.DataFrame: Output features
    """

    features_subset = features.iloc[:, 3:]  # Skipping first three (metadata) columns
    y = (features['label'] == 'N').to_numpy(
        dtype=int
    )  # 0 = subcrater S; 1 = subcrater N

    print('Applying TSFRESH feature selection...')
    features_filtered = select_features(features_subset, y, **selection_kwargs)
    print(
        f'# of features reduced from {features_subset.shape[1]} to {features_filtered.shape[1]}'
    )

    return pd.concat([features.iloc[:, :3], features_filtered], axis='columns')


def balance_classes(features, random_state=None, verbose=True):
    """Function to adjust for class imbalance by down-sampling the majority class.

    See https://elitedatascience.com/imbalanced-classes for more info. This function
    operates on the "event" level, not on the station level. I.e. the five stations
    associated with a single event (i.e. having a shared origin time) are not broken
    apart by the sampling.

    Args:
        features (pandas.DataFrame): Input features (must have "label" and "station"
            columns)
        random_state: Passed on to resample(), see https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        verbose (bool): Toggle printing info about balancing

    Returns:
        pandas.DataFrame: Output features
    """

    # Perform the balancing
    class_counts_before = features.label.value_counts()
    if verbose:
        print('Before:\n' + class_counts_before.to_string())
    dominant_label = class_counts_before.index[class_counts_before.argmax()]
    majority = features[features.label == dominant_label]
    minority = features[features.label != dominant_label]

    # When resampling, use only the features associated with a single station - this
    # treats things on an event level
    unique_stations = features.station.unique()
    majority_single_station = majority[majority.station == unique_stations[0]]
    majority_single_station_downsampled = resample(
        majority_single_station,
        replace=False,  # Sample w/o replacement
        n_samples=int(
            minority.shape[0] / unique_stations.size
        ),  # Match number of EVENTS in minority class
        random_state=random_state,
    )

    # Now use the event times that remain to index the full features
    majority_downsampled = majority[
        majority.time.astype(float).isin(
            majority_single_station_downsampled.time.astype(float)
        )
    ]

    # Check that each station has the same number of examples (for each subcrater)
    assert (
        len(
            set(
                minority.station.value_counts().tolist()
                + majority_downsampled.station.value_counts().tolist()
            )  # Takes the unique value; there should just only 1!
        )
        == 1
    )

    features_downsampled = pd.concat([majority_downsampled, minority])

    if verbose:
        # Print what action was taken
        class_counts_after = features_downsampled.label.value_counts()
        print('After:\n' + class_counts_after.to_string())
        num_removed = (class_counts_before - class_counts_after)[dominant_label]
        print(f'({num_removed} subcrater {dominant_label} examples removed)')

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
    y = (features['label'] == 'N').to_numpy(
        dtype=int
    )  # 0 = subcrater S; 1 = subcrater N

    return X, y


def plot_confusion(clf, X_test, y_test, title=None):
    """Shallow wrapper around plot_confusion_matrix().

    See
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
    for more info. This function basically tweaks the standard plot for the Yasur
    context.

    Warning:
        Double-check that the colored "S" and "N" labels are set in boldface. The
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
        display_labels=['S', 'N'],  # Since 0 = subcrater S; 1 = subcrater N
        cmap='Greys',
        normalize='true',  # 'true' means the diagonal contains the TPR and TNR
        values_format='.0%',  # Format as integer percent
        colorbar=False,  # Don't add colorbar
    )
    cm.im_.set_clim(0, 1)  # For easier comparison between plots
    fig = cm.figure_
    ax = fig.axes[0]
    ax.set_xlabel(ax.get_xlabel().replace('label', 'subcrater'))
    ax.set_ylabel(ax.get_ylabel().replace('label', 'subcrater'))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('bold')
        label.set_color(os.environ[f'SUBCRATER_{label.get_text()}'])
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


def _plot_correlation(features, dendro_ax, corr_ax, show_names=False, plot_thresh=None):
    """Called by remove_correlated_features().

    Basically a function version of
    https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#handling-multicollinear-features

    show_names=True is slow for large numbers of features!
    """

    X, _ = format_scikit(features)

    # Form correlation matrix and ensure it's symmetric
    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # Plot dendrogram of clustered features
    dendro = hierarchy.dendrogram(
        dist_linkage,
        labels=features.columns[3:],
        no_labels=False if show_names else True,
        ax=dendro_ax,
        leaf_rotation=90,
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))
    if plot_thresh:
        dendro_ax.axhline(plot_thresh, color='black', lw=2, zorder=5)
        dendro_ax.text(
            dendro_ax.get_xlim()[1],
            plot_thresh,
            f'  {plot_thresh}',
            weight='bold',
            va='center',
            ha='left',
        )

    # Plot absolute value of correlation between features
    corr_ax.imshow(
        np.abs(corr[dendro['leaves'], :][:, dendro['leaves']]),
        cmap='Greys',
        vmin=0,
        vmax=1,
    )
    if show_names:
        corr_ax.set_xticks(dendro_idx)
        corr_ax.set_yticks(dendro_idx)
        corr_ax.set_xticklabels(dendro['ivl'], rotation='vertical')
        corr_ax.set_yticklabels(dendro['ivl'])
    corr_ax.set_title(f'{X.shape[1]} features')

    return dist_linkage


def remove_correlated_features(
    features, thresh, show_before_names=False, show_after_names=False
):
    """Remove features that are correlated by thresholding clusters.

    Args:
        features (pandas.DataFrame): Input features
        thresh (int or float): Threshold below which clusters are collapsed to their
            representative feature; higher value means fewer output features
        show_before_names (bool): Toggles showing feature names for "before" features
        show_after_names (bool):  Toggles showing feature names for "after" features

    Returns:
        pandas.DataFrame: Output features
    """

    fig, (before_row, after_row) = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Plot "before" features (and threshold) while obtaining distances
    dist_linkage = _plot_correlation(
        features, *before_row, plot_thresh=thresh, show_names=show_before_names
    )

    # If thresh is not provided (or 0) set it to negative to ensure no features removed
    if not thresh:
        thresh = -1

    # Threshold and select features
    cluster_ids = hierarchy.fcluster(dist_linkage, thresh, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    # Form output DataFrame
    features_reduced = features[
        features.columns[:3].tolist() + features.columns[3:][selected_features].tolist()
    ]

    # Plot "after" features
    _ = _plot_correlation(features_reduced, *after_row, show_names=show_after_names)

    before_row[0].set_title('Before')
    after_row[0].set_title('After')
    fig.suptitle(features.attrs['filename'], weight='bold')

    fig.tight_layout()
    fig.show()

    print(
        f'# features reduced from {len(features.columns[3:])} to {len(features_reduced.columns[3:])}'
    )

    return features_reduced

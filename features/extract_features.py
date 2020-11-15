from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read
from scipy import stats
from scipy.signal import welch

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

#%% Extract features for a single station

FFT_WIN_DUR = 10  # [s]

STATION = 'YIF2'  # Station to extract features for

# Initiate DataFrame of extracted features
features = pd.DataFrame(
    columns=['label', 'td_std', 'td_skewness', 'td_kurtosis', 'fd_peak']
)

# Iterate over all labeled waveform files
for file in sorted(labeled_wf_dir.glob('label_???.pkl')):

    # Read in
    print(f'Reading {file}')
    st = read(str(file)).select(station=STATION)

    # Process
    st.remove_response()
    st.taper(0.01)
    # st.filter('bandpass', freqmin=0.2, freqmax=4, zerophase=True)
    st.normalize()

    # Calculate features, append to DataFrame
    for tr in st:

        # Transform to frequency domain
        fs = tr.stats.sampling_rate
        nperseg = int(FFT_WIN_DUR * fs)  # Samples
        nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 1)  # Pad FFT
        f, psd = welch(tr.data, fs, nperseg=nperseg, nfft=nfft)

        info = dict(
            label=tr.stats.vent,
            td_std=np.std(tr.data),
            td_skewness=stats.skew(tr.data),
            td_kurtosis=stats.kurtosis(tr.data),
            fd_peak=f[np.argmax(psd)],  # [Hz]
        )
        features = features.append(info, ignore_index=True)

#%% Plot two features against each other as a scatter plot

X_AXIS_FEATURE = 'td_skewness'
Y_AXIS_FEATURE = 'td_kurtosis'

colors = ['blue' if label == 'A' else 'red' for label in features['label']]

fig, ax = plt.subplots()
ax.scatter(
    features[X_AXIS_FEATURE],
    features[Y_AXIS_FEATURE],
    edgecolors=colors,
    facecolors='none',
)
ax.set_xlabel(X_AXIS_FEATURE)
ax.set_ylabel(Y_AXIS_FEATURE)
ax.set_title(f'{STATION}, {features.shape[0]} waveforms')

# Add legend
ax.scatter([], [], edgecolors='blue', facecolors='none', label='Vent A')
ax.scatter([], [], edgecolors='red', facecolors='none', label='Vent C')
ax.legend()

fig.show()

#%% Histograms of each feature

feature_names = features.columns[1:]  # Skip first column since it's the label

NCOLS = 3  # Number of subplot columns

fig, axes = plt.subplots(nrows=int(np.ceil(len(feature_names) / NCOLS)), ncols=NCOLS)

for ax, feature in zip(axes.flatten(), feature_names):
    ax.hist(features[feature], bins=50)
    ax.set_title(feature)

# Remove empty subplots, if any
for ax in axes.flatten():
    if not ax.has_data():
        fig.delaxes(ax)

fig.suptitle(f'{STATION}, {features.shape[0]} waveforms')
fig.tight_layout()
fig.show()

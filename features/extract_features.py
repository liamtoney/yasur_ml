import os
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
    columns=[
        'label',
        'td_std',
        'td_skewness',
        'td_kurtosis',
        'fd_std',
        'fd_skewness',
        'fd_kurtosis',
        'fd_q1',
        'fd_q2',
        'fd_q3',
        'fd_peak',
    ]
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

        # Get CSD, normalize
        csd = np.cumsum(psd)
        csd_norm = csd / csd.max()
        psd_norm = psd / psd.max()

        # Find quartiles
        quartiles = {}
        for quartile in 0.25, 0.5, 0.75:
            idx = (np.abs(csd_norm - quartile)).argmin()
            quartiles[quartile] = f[idx]

        info = dict(
            label=tr.stats.vent,
            td_std=np.std(tr.data),
            td_skewness=stats.skew(tr.data),
            td_kurtosis=stats.kurtosis(tr.data),
            fd_std=np.std(psd_norm),
            fd_skewness=stats.skew(psd_norm),
            fd_kurtosis=stats.kurtosis(psd_norm),
            fd_q1=quartiles[0.25],  # [Hz]
            fd_q2=quartiles[0.5],  # [Hz]
            fd_q3=quartiles[0.75],  # [Hz]
            fd_peak=f[np.argmax(psd)],  # [Hz]
        )
        features = features.append(info, ignore_index=True)

#%% Plot two features against each other as a scatter plot

X_AXIS_FEATURE = 'td_skewness'
Y_AXIS_FEATURE = 'td_kurtosis'

colors = [os.environ[f'VENT_{label}'] for label in features.label]

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
ax.scatter([], [], edgecolors=os.environ['VENT_A'], facecolors='none', label='Vent A')
ax.scatter([], [], edgecolors=os.environ['VENT_C'], facecolors='none', label='Vent C')
ax.legend()

fig.show()

#%% Histograms of each feature

feature_names = features.columns[1:]  # Skip first column since it's the label

SPLIT_BY_LABEL = True

NCOLS = 3  # Number of subplot columns
NBINS = 50  # Number of histogram bins

fig, axes = plt.subplots(
    nrows=int(np.ceil(len(feature_names) / NCOLS)), ncols=NCOLS, figsize=(8, 10)
)

for ax, feature in zip(axes.flatten(), feature_names):
    if SPLIT_BY_LABEL:
        ax.hist(
            features[features.label == 'A'][feature],
            bins=NBINS,
            color=os.environ['VENT_A'],
        )
        ax.hist(
            features[features.label == 'C'][feature],
            bins=NBINS,
            color=os.environ['VENT_C'],
        )
    else:
        ax.hist(features[feature], bins=NBINS, color='grey')
    ax.set_title(feature)

# Remove empty subplots, if any
for ax in axes.flatten():
    if not ax.has_data():
        fig.delaxes(ax)

fig.suptitle(f'{STATION}, {features.shape[0]} waveforms')
fig.tight_layout()
fig.show()

#%% (OPTIONAL) Example plot of frequency domain features

fig, axes = plt.subplots(
    ncols=2, figsize=(6, 3), sharey=True, gridspec_kw=dict(width_ratios=[0.7, 0.3])
)

axes[0].plot(f, psd_norm, label='PSD')
axes[0].plot(f, csd_norm, '--', color='black', label='CSD')
axes[0].scatter(
    [v for v in quartiles.values()],
    [k for k in quartiles.keys()],
    color='black',
    label='Quartiles',
    zorder=100,
)
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('Normalized PSD/CSD')
axes[0].autoscale(tight=True)
axes[0].legend()

axes[1].hist(psd_norm, bins=20, orientation='horizontal')
axes[1].set_xlabel('PSD counts')
axes[1].tick_params(labelleft=False)
axes[1].autoscale(tight=True)

fig.tight_layout()
fig.show()

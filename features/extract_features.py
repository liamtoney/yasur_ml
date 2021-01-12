from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read
from scipy import stats
from scipy.signal import welch
from tsfresh import extract_features

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

# Station to extract features for, use None for all stations
STATION = 'YIF2'

# Toggle bandpass filtering of data
FILTER = True

# Bandpass filter corners [Hz]
FREQMIN = 0.2
FREQMAX = 4

#%% Option 1: Manual feature engineering

FFT_WIN_DUR = 5  # [s]

# Initialize DataFrame of extracted features
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
    if STATION:
        st = read(str(file)).select(station=STATION)  # Use only STATION
    else:
        st = read(str(file))  # Use all stations

    # Process
    st.remove_response()
    st.taper(0.01)
    if FILTER:
        st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
    st.normalize()

    # Calculate features, append to DataFrame
    for tr in st:

        # Transform to frequency domain
        fs = tr.stats.sampling_rate
        nperseg = int(FFT_WIN_DUR * fs)  # Samples
        nfft = np.power(2, int(np.ceil(np.log2(nperseg))) + 3)  # Pad FFT
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

# Save as CSV
if STATION:
    filename = f'{STATION}_features.csv'
else:
    filename = 'features.csv'
if FILTER:
    filename = filename.replace('.csv', '_filtered.csv')
features.to_csv(WORKING_DIR / 'features' / 'csv' / filename, index=False)

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

#%% Option 2: TSFRESH

# Initialize DataFrame of extracted features
features = pd.DataFrame()

# Iterate over all labeled waveform files
for file in sorted(labeled_wf_dir.glob('label_???.pkl')):

    # Read in
    print(f'Reading {file}')
    if STATION:
        st = read(str(file)).select(station=STATION)  # Use only STATION
    else:
        st = read(str(file))  # Use all stations

    # Process
    st.remove_response()
    st.taper(0.01)
    if FILTER:
        st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
    st.normalize()

    # Put into TSFRESH format
    timeseries = pd.DataFrame()
    labels = []
    for i, tr in enumerate(st):
        id = pd.Series(np.ones(tr.stats.npts, dtype=int) * i)
        time = pd.Series(tr.times())
        value = pd.Series(tr.data)
        timeseries = pd.concat(
            [timeseries, pd.DataFrame(dict(id=id, time=time, x=value))],
            ignore_index=True,
        )
        labels.append(tr.stats.vent)

    # Calculate features
    extracted_features = extract_features(
        timeseries, column_id='id', column_sort='time'
    )

    # Tweak and append to main DataFrame
    extracted_features.insert(0, column='label', value=labels)
    extracted_features.columns = [
        column.split('__', 1)[-1] for column in extracted_features.columns
    ]
    features = pd.concat([features, extracted_features], ignore_index=True)

# Save as CSV
if STATION:
    filename = f'{STATION}_features_tsfresh.csv'
else:
    filename = 'features_tsfresh.csv'
if FILTER:
    filename = filename.replace('.csv', '_filtered.csv')
features.to_csv(WORKING_DIR / 'features' / 'csv' / filename, index=False)

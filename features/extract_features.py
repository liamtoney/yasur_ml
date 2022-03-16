#!/usr/bin/env python

import os
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read
from scipy import stats
from scipy.signal import welch
from tsfresh import extract_features

# Define project directory
WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

# Toggle bandpass filtering of data
FILTER = True

# Bandpass filter corners [Hz]
FREQMIN = 0.2
FREQMAX = 4

# Toggle randomly shifting travel times via np.roll()
ROLL = True

# Specify 'manual' or 'tsfresh' feature set
SET_TYPE = 'tsfresh'


def manual_extractor(st):
    """
    Extracts 10 "hand-picked" manually defined features.
    """

    FFT_WIN_DUR = 5  # [s]

    # Initialize DataFrame of extracted features
    features_st = pd.DataFrame()

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

        features_tr = pd.DataFrame(
            dict(
                station=tr.stats.station,
                time=str(tr.stats.event_info.origin_time),  # Origin time from catalog
                label=tr.stats.subcrater,
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
            ),
            index=[0],
        )
        features_st = pd.concat([features_st, features_tr], ignore_index=True)

    return features_st


def tsfresh_extractor(st):
    """
    Extracts 700+ features automatically, leveraging TSFRESH.
    """

    # Put into TSFRESH format
    timeseries = pd.DataFrame()
    labels = []
    otimes = []  # Origin time from catalog
    stations = []
    for i, tr in enumerate(st):
        id = pd.Series(np.ones(tr.stats.npts, dtype=int) * i)
        time = pd.Series(tr.times())
        value = pd.Series(tr.data)
        timeseries = pd.concat(
            [timeseries, pd.DataFrame(dict(id=id, time=time, x=value))],
            ignore_index=True,
        )
        labels.append(tr.stats.subcrater)
        otimes.append(str(tr.stats.event_info.origin_time))
        stations.append(tr.stats.station)

    # Calculate features
    features_st = extract_features(timeseries, column_id='id', column_sort='time')

    # Append label, time, and station info to main DataFrame (last appended = first col)
    features_st.insert(0, column='label', value=labels)
    features_st.insert(0, column='time', value=otimes)
    features_st.insert(0, column='station', value=stations)

    # Slightly shorten the very long TSFRESH feature [= column] names
    features_st.columns = [column.split('__', 1)[-1] for column in features_st.columns]

    return features_st


# Choose specified feature extractor
if SET_TYPE == 'manual':
    feature_extractor = manual_extractor
elif SET_TYPE == 'tsfresh':
    feature_extractor = tsfresh_extractor
else:
    raise ValueError('Invalid SET_TYPE!')

# Initialize main DataFrame of extracted features
features = pd.DataFrame()

# Iterate over all labeled waveform files
for file in sorted(labeled_wf_dir.glob('label_???.pkl')):

    # Read in
    print(f'Reading {file}')
    st = read(str(file))

    # Pre-process prior to extraction
    st.remove_response()
    st.taper(0.01)
    if FILTER:
        st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
    if ROLL:
        # Randomly shuffle waveforms to remove travel time biases
        for tr in st:
            tr.data = np.roll(tr.data, np.random.randint(tr.data.size))
    st.normalize()

    # Extract features
    features_st = feature_extractor(st)

    # Append to main DataFrame
    features = pd.concat([features, features_st], ignore_index=True)

# Save as Feather file
filename = f'{SET_TYPE}.feather'
if FILTER:
    filename = filename.replace('.feather', '_filter.feather')
if ROLL:
    filename = filename.replace('.feather', '_roll.feather')
features.to_feather(WORKING_DIR / 'features' / 'feather' / filename)

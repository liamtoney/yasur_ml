from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read
from scipy import stats

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

#%% Extract features for a single station

STATION = 'YIF2'  # Station to extract features for

# Initiate DataFrame of extracted features
features = pd.DataFrame(columns=['label', 'td_std', 'td_skewness', 'td_kurtosis'])

# Iterate over all labeled waveform files
for file in sorted(labeled_wf_dir.glob('label_???.pkl')):

    # Read in
    print(f'Reading {file}')
    st = read(str(file)).select(station=STATION)

    # Process
    st.remove_response()
    st.taper(0.01)
    st.filter('bandpass', freqmin=0.2, freqmax=4, zerophase=True)
    st.normalize()

    # Calculate features, append to DataFrame
    for tr in st:
        info = dict(
            label=tr.stats.vent,
            td_std=np.std(tr.data),
            td_skewness=stats.skew(tr.data),
            td_kurtosis=stats.kurtosis(tr.data),
        )
        features = features.append(info, ignore_index=True)

#%% Plot features

colors = ['blue' if label == 'A' else 'red' for label in features['label']]

fig, ax = plt.subplots()
ax.scatter(
    features['td_skewness'],
    features['td_kurtosis'],
    edgecolors=colors,
    facecolors='none',
)
ax.set_xlabel('T.D. skewness')
ax.set_ylabel('T.D. kurtosis')
ax.set_title(f'{STATION}, {features.shape[0]} waveforms')

# Add legend
ax.scatter([], [], edgecolors='blue', facecolors='none', label='Vent A')
ax.scatter([], [], edgecolors='red', facecolors='none', label='Vent C')
ax.legend()

fig.show()

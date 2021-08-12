#!/usr/bin/env python

import copy
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import read

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

# Read in a single Stream from the labeled waveforms to get metadata etc.
st = read(str(labeled_wf_dir / 'label_000.pkl'))
STATIONS = sorted(np.unique([tr.stats.station for tr in st]))
NPTS = st[0].stats.npts

# Toggle filtering the waveforms prior to stacking
FILTER = False

# Define filename for saved traces
filter_str = '_filtered' if FILTER else ''
pickle_filename = WORKING_DIR / 'plot_scripts' / f'traces_dict{filter_str}.pkl'

# Only read in the files and stack if we NEED to, since this takes a while!
if not pickle_filename.exists():

    # Construct dictionary to hold traces for each vent and each station
    station_dict = {}
    for station in STATIONS:
        # This initial NaN trace must be removed later
        station_dict[station] = np.full(NPTS, np.nan)
    traces = dict(A=copy.deepcopy(station_dict), C=copy.deepcopy(station_dict))

    # Iterate over all labeled waveform files
    for file in sorted(labeled_wf_dir.glob('label_???.pkl')):

        # Read in file
        print(f'Reading {file}')
        st = read(str(file))

        # Process
        st.remove_response()

        if FILTER:
            st.filter('bandpass', freqmin=0.2, freqmax=4, zerophase=True)

        # Loop over all stations
        for station in STATIONS:

            # Add traces
            for tr in st.select(station=station):
                if tr.stats.vent == 'A':
                    traces['A'][station] = np.vstack((traces['A'][station], tr.data))
                else:
                    traces['C'][station] = np.vstack((traces['C'][station], tr.data))

    # Write file
    with pickle_filename.open('wb') as f:
        pickle.dump(traces, f)

else:

    # Just read in file
    with pickle_filename.open('rb') as f:
        traces = pickle.load(f)

#%% Plot stacks

fig, axes = plt.subplots(nrows=len(STATIONS), ncols=2, sharex=True, sharey=True)

for vent, axes_col in zip(traces.keys(), axes.T):
    axes_col[0].set_title(f'Vent {vent}')
    for station, ax in zip(traces[vent].keys(), axes_col):
        vs_traces = traces[vent][station][1:, :]  # Removing the first row of NaNs here

        # Stack up all waveforms with original amps (in Pa)
        stack = vs_traces.sum(axis=0)

        ax.plot(st[0].times(), stack / stack.max())  # Normalizing here
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 5)
        ax.set_yticks([])
        ax.set_ylabel(station)

    axes_col[-1].set_xlabel('Time (s)')

fig.tight_layout()
fig.show()

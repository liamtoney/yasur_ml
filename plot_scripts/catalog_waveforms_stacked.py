#!/usr/bin/env python

import copy
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, Trace, read

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
FMIN = 0.2
FMAX = 4

# Common figsize
FIGSIZE = (7, 7)

SAVE = False

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
            st.filter('bandpass', freqmin=FMIN, freqmax=FMAX, zerophase=True)

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

fig, axes = plt.subplots(
    nrows=len(STATIONS), ncols=2, sharex=True, sharey=False, figsize=FIGSIZE
)

for vent, axes_col in zip(traces.keys(), axes.T):
    axes_col[0].set_title(f'Vent {vent}')
    color = os.environ[f'VENT_{vent}']
    for station, ax in zip(traces[vent].keys(), axes_col):
        vs_traces = traces[vent][station][1:, :]  # Removing the first row of NaNs here

        med = np.percentile(vs_traces, 50, axis=0)
        ax.plot(st[0].times(), med, color=color, zorder=5)
        ax.fill_between(
            st[0].times(),
            np.percentile(vs_traces, 25, axis=0),
            np.percentile(vs_traces, 75, axis=0),
            color=color,
            linewidth=0,
            alpha=0.3,
        )
        med_max = np.abs(med).max()
        ax.set_ylim(-2 * med_max, 2 * med_max)  # Normalizing by median
        ax.set_xlim(0, 5)
        ax.set_yticks([])
        ax.set_ylabel(station)

    axes_col[-1].set_xlabel('Time (s)')

fig.suptitle('Stacked waveforms')
fig.tight_layout()
fig.show()

if SAVE:
    fig.savefig(
        '/Users/ldtoney/Downloads/stacked_waveforms.png', bbox_inches='tight', dpi=300
    )

#%% Plot GFs

# Define GF directory
gf_dir = WORKING_DIR / 'label' / 'gf'


# Function to read in GF text files from David and produce a Stream w/ timing info
def read_gf(filename, vent):
    with open(filename) as f:
        header = f.readline().strip('#').strip().split(' ')
    gf_df = pd.read_csv(filename, delim_whitespace=True, comment='#', names=header)
    delta = gf_df.tvec[1] - gf_df.tvec[0]
    traces = [
        Trace(
            data=gf_df[station].values,
            header=dict(station=station, sampling_rate=1 / delta, vent=vent),
        )
        for station in gf_df.columns[:-1]
    ]
    return Stream(traces)


# Read in text files
gf_A = read_gf(gf_dir / 'VentA_gf.txt', vent='A')
gf_C = read_gf(gf_dir / 'VentC_gf.txt', vent='C')

fig, axes = plt.subplots(
    nrows=len(STATIONS), ncols=2, sharex=True, sharey=True, figsize=FIGSIZE
)

for st, vent, axes_col in zip([gf_A, gf_C], traces.keys(), axes.T):

    # Remove YIF6
    for tr in st.select(station='YIF6'):
        st.remove(tr)

    axes_col[0].set_title(f'Vent {st[0].stats.vent}')
    color = os.environ[f'VENT_{st[0].stats.vent}']

    # Process the Stream
    stp = st.copy()
    if FILTER:
        stp.filter('bandpass', freqmin=FMIN, freqmax=FMAX, zerophase=True)
    stp.normalize()

    for tr, ax in zip(stp, axes_col):
        ax.plot(tr.times(), tr.data, color=color)
        ax.set_ylim(-2, 2)  # For comparison with median of stack
        ax.set_xlim(0, 5)
        ax.set_yticks([])
        ax.set_ylabel(tr.stats.station)

    axes_col[-1].set_xlabel('Time (s)')

fig.suptitle('GFs')
fig.tight_layout()
fig.show()

if SAVE:
    fig.savefig('/Users/ldtoney/Downloads/GFs.png', bbox_inches='tight', dpi=300)

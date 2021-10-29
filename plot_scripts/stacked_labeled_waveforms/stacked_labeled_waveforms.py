#!/usr/bin/env python

import copy
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from obspy import read

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

FONT_SIZE = 12  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Read in a single Stream from the labeled waveforms to get metadata etc.
st = read(str(labeled_wf_dir / 'label_000.pkl'))
STATIONS = sorted(np.unique([tr.stats.station for tr in st]))
NPTS = st[0].stats.npts

# Toggle filtering the waveforms prior to stacking
FILTER = True
FMIN = 0.2
FMAX = 4

# Define filename for saved traces
filter_str = '_filtered' if FILTER else ''
pickle_filename = (
    WORKING_DIR
    / 'plot_scripts'
    / 'stacked_labeled_waveforms'
    / f'traces_dict{filter_str}.pkl'
)

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
        st.taper(0.01)
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
    nrows=len(STATIONS), ncols=2, sharex=True, sharey=False, figsize=(7, 7)
)

for vent, axes_col in zip(traces.keys(), axes.T):
    axes_col[0].set_title(
        f'Subcrater {"S" if vent == "A" else "N"}', fontsize=FONT_SIZE
    )
    color = os.environ[f'VENT_{vent}']
    for station, ax in zip(traces[vent].keys(), axes_col):
        vs_traces = traces[vent][station][1:, :]  # Removing the first row of NaNs here

        med = np.percentile(vs_traces, 50, axis=0)
        ax.plot(
            st[0].times(),
            med,
            color=color,
            zorder=5,
            solid_capstyle='round',
            clip_on=False,
        )
        ax.fill_between(
            st[0].times(),
            np.percentile(vs_traces, 25, axis=0),
            np.percentile(vs_traces, 75, axis=0),
            color=color,
            linewidth=0,
            alpha=0.3,
            clip_on=False,
        )
        med_max = np.abs(med).max()
        scale = 2.5
        ax.set_ylim(-scale * med_max, scale * med_max)  # Normalizing by median
        ax.set_xlim(0, 5)
        ax.set_yticks([])

        # Label stations (only need to do this for one column)
        if vent == 'A':
            trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
            ax.text(0.5, 0.49, station, va='center', ha='center', transform=trans)

        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes_col[:-1]:
        ax.tick_params(axis='x', which='both', bottom=False)

    axes_col[-1].set_xlabel('Time (s)')
    axes_col[-1].spines['bottom'].set_visible(True)
    axes_col[-1].xaxis.set_tick_params(direction='in', pad=5)

fig.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0.25)

fig.show()

# fig.savefig(Path(os.environ['YASUR_FIGURE_DIR']) / 'stacked_labeled_waveforms.png', bbox_inches='tight', dpi=300)

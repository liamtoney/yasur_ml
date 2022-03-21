#!/usr/bin/env python

import copy
import os
import pickle
import subprocess
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

# Define project directory
WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

FONT_SIZE = 14  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# [s] Rolling window duration
WINDOW = 60 * 60

# Read in features file, process into a catalog by subsetting to just one station and
# removing the unneeded features columns
catalog = pd.read_feather(
    WORKING_DIR / 'features' / 'feather' / 'manual_filter_roll.feather'
)
catalog = catalog[catalog.station == 'YIF1'][['time', 'label']]
catalog.time = [UTCDateTime(t) for t in catalog.time]

# Start and end on whole hours
t_start = UTCDateTime('2016-07-27T05')
t_end = UTCDateTime('2016-08-01T22')

# Form array of UTCDateTimes
t_vec = [t_start + t for t in np.arange(0, (t_end - t_start), WINDOW)]

# In moving windows, get counts of subcrater S and subcrater N
fraction_S = []
fraction_N = []
for t in t_vec:
    catalog_hr = catalog[(catalog.time >= t) & (catalog.time < t + WINDOW)]
    vcounts = catalog_hr.label.value_counts()
    if hasattr(vcounts, 'S'):
        fraction_S.append(vcounts.S)
    else:
        fraction_S.append(0)
    if hasattr(vcounts, 'N'):
        fraction_N.append(vcounts.N)
    else:
        fraction_N.append(0)

# Load in a single station's data and process (takes a while)
tr = read(str(WORKING_DIR / 'data' / '3E_YIF1-5_50hz.pkl')).select(station='YIF3')[0]
tr.remove_response()
tr.filter('bandpass', freqmin=0.2, freqmax=4, zerophase=True)

# Read in a single Stream from the labeled waveforms to get metadata etc.
st = read(str(labeled_wf_dir / 'label_000.pkl'))
STATIONS = sorted(np.unique([tr.stats.station for tr in st]))
NPTS = st[0].stats.npts

# Toggle filtering the waveforms prior to stacking
FILTER = True
FMIN = 0.2
FMAX = 4

# Set up figure
fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(
    nrows=2 + len(STATIONS) + 2,  # 2 for top two panels, 2 for spacers
    ncols=2,
    height_ratios=[1, 0.2, 1, 0.4] + list(2.5 / len(STATIONS) * np.ones(len(STATIONS))),
)

#%% (a), (b)

ax1 = fig.add_subplot(gs[0, :])
space1 = fig.add_subplot(gs[1, :])
space1.set_visible(False)
ax2 = fig.add_subplot(gs[2, :], sharex=ax1)
space2 = fig.add_subplot(gs[3, :])
space2.set_visible(False)

# Subplot 1: Waveform
tr_trim = tr.copy().trim(t_vec[0] + (WINDOW / 2), t_vec[-1] + (WINDOW / 2))
ax1.plot(
    tr_trim.times('matplotlib'),
    tr_trim.data,
    linewidth=0.5,
    color='black',
    clip_on=False,
    solid_capstyle='round',
)
ax1.set_ylabel('Pressure (Pa)')
ax1.set_ylim(-500, 500)
ax1.yaxis.set_major_locator(plt.MultipleLocator(250))

ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

# Subplot 2: Stacked area plot
t_vec_mpl = [(t + (WINDOW / 2)).matplotlib_date for t in t_vec]  # Center in window!
ax2.stackplot(
    t_vec_mpl,
    fraction_S,
    fraction_N,
    colors=(os.environ['SUBCRATER_S'], os.environ['SUBCRATER_N']),
    labels=('S subcrater', 'N subcrater'),
    clip_on=False,
)
ax2.set_ylabel('# of labeled events')
ax2.set_ylim(0, 70)
ax2.yaxis.set_minor_locator(plt.MultipleLocator(10))

# Remove spines
for side in 'bottom', 'top', 'right':
    ax1.spines[side].set_visible(False)
for side in 'top', 'right':
    ax2.spines[side].set_visible(False)

# Overall x-axis formatting
ax2.set_xlim(t_vec_mpl[0], t_vec_mpl[-1])  # Bounds of the area plot
loc = ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%-d %B'))
ax2.xaxis.set_minor_locator(mdates.HourLocator(range(0, 24, 12)))

# Add legend
ax2.legend(loc='lower right', framealpha=0.6, fancybox=False, edgecolor='none')

# Make connected gridlines
for ax in ax1, ax2:
    ax.patch.set_alpha(0)
grid_ax = fig.add_subplot(gs[:3, :], zorder=-1, sharex=ax2)
for spine in grid_ax.spines.values():
    spine.set_visible(False)
grid_ax.tick_params(
    which='both',
    left=False,
    labelleft=False,
    bottom=False,
    labelbottom=False,
)
grid_ax.grid(which='both', axis='x', linestyle=':', alpha=0.5)

#%% (setup)

# Define filename for saved traces
filter_str = '_filtered' if FILTER else ''
pickle_filename = (
    WORKING_DIR / 'plot_scripts' / 'catalog_waveforms' / f'traces_dict{filter_str}.pkl'
)

# Only read in the files and stack if we NEED to, since this takes a while!
if not pickle_filename.exists():

    # Construct dictionary to hold traces for each subcrater and each station
    station_dict = {}
    for station in STATIONS:
        # This initial NaN trace must be removed later
        station_dict[station] = np.full(NPTS, np.nan)
    traces = dict(S=copy.deepcopy(station_dict), N=copy.deepcopy(station_dict))

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
                if tr.stats.subcrater == 'S':
                    traces['S'][station] = np.vstack((traces['S'][station], tr.data))
                else:
                    traces['N'][station] = np.vstack((traces['N'][station], tr.data))

    # Write file
    with pickle_filename.open('wb') as f:
        pickle.dump(traces, f)

else:

    # Just read in file
    with pickle_filename.open('rb') as f:
        traces = pickle.load(f)

#%% (c), (d)

# Awful, but works

ax11 = fig.add_subplot(gs[4, 0])
ax21 = fig.add_subplot(gs[5, 0], sharex=ax11)
ax31 = fig.add_subplot(gs[6, 0], sharex=ax11)
ax41 = fig.add_subplot(gs[7, 0], sharex=ax11)
ax51 = fig.add_subplot(gs[8, 0], sharex=ax11)

ax12 = fig.add_subplot(gs[4, 1])
ax22 = fig.add_subplot(gs[5, 1], sharex=ax12)
ax32 = fig.add_subplot(gs[6, 1], sharex=ax12)
ax42 = fig.add_subplot(gs[7, 1], sharex=ax12)
ax52 = fig.add_subplot(gs[8, 1], sharex=ax12)

axes = np.array([[ax11, ax21, ax31, ax41, ax51], [ax12, ax22, ax32, ax42, ax52]]).T

for i, (subcrater, axes_col) in enumerate(zip(traces.keys(), axes.T)):
    color = os.environ[f'SUBCRATER_{subcrater}']
    for station, ax in zip(traces[subcrater].keys(), axes_col):
        vs_traces = traces[subcrater][station][
            1:, :
        ]  # Removing the first row of NaNs here

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
        if subcrater == 'S':
            trans = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
            X_LOC = 0.5276  # TODO: MANUAL ADJUSTMENT TO TRULY CENTER
            ax.text(X_LOC, 0.49, station, va='center', ha='center', transform=trans)

        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes_col[:-1]:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # Add gridlines
    for ax in axes_col:
        ax.patch.set_alpha(0)
    grid_ax = fig.add_subplot(gs[4:, i], zorder=-1, sharex=axes_col[-1])

    for spine in grid_ax.spines.values():
        spine.set_visible(False)
    grid_ax.tick_params(
        which='both',
        left=False,
        labelleft=False,
        bottom=False,
        labelbottom=False,
    )
    for x in np.arange(1, 5):
        grid_ax.axvline(
            x=x,
            linestyle=':',
            zorder=-1,
            color=plt.rcParams['grid.color'],
            linewidth=plt.rcParams['grid.linewidth'],
            alpha=0.5,
            clip_on=False,
        )

    axes_col[-1].set_xlabel('Time (s)')
    axes_col[-1].spines['bottom'].set_visible(True)

# Plot (a), (b), (c), and (d) tags
label_kwargs = dict(
    ha='left',
    va='top',
    weight='bold',
    fontsize=18,
)
ax1.text(0.01, 1.03, s='(a)', transform=ax1.transAxes, **label_kwargs)
ax2.text(0.01, 1.03, s='(b)', transform=ax2.transAxes, **label_kwargs)
axes[0, 0].text(
    0.01,
    1,
    s='(c)',
    transform=transforms.blended_transform_factory(ax1.transAxes, axes[0, 0].transAxes),
    **label_kwargs,
)
axes[0, 1].text(0.025, 1, s='(d)', transform=axes[0, 1].transAxes, **label_kwargs)

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0.2)
fig.show()

_ = subprocess.run(['open', os.environ['YASUR_FIGURE_DIR']])

# fig.savefig(Path(os.environ['YASUR_FIGURE_DIR']).expanduser().resolve() / 'catalog_waveforms.png', bbox_inches='tight', dpi=300)

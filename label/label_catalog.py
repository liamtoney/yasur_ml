"""
Read in CSV file corresponding to a catalog, define the peaks of the histograms around
each vent, and label events based upon proximity to these peaks.
"""

import json
import os
from pathlib import Path

import colorcet as cc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm
from matplotlib.ticker import PercentFormatter
from obspy import Stream, UTCDateTime, read
from rtm import define_grid, produce_dem

# Toggle plotting
PLOT = False

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Define which catalog to label
catalog_csv = WORKING_DIR / 'label' / 'catalogs' / 'height_4_spacing_30_agc_60.csv'

# [m] Maximum distance from vent location estimate to a given event
MAX_RADIUS = 40

# Read in entire catalog to pandas DataFrame
df = pd.read_csv(catalog_csv)
df.t = [UTCDateTime(t) for t in df.t]

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# Convert vent midpoint to UTM
x_0, y_0, *_ = utm.from_latlon(VENT_LOCS['midpoint'][1], VENT_LOCS['midpoint'][0])

# Need to define this for proper histogram binning (MUST MATCH build_catalog.py)
grid = define_grid(
    lon_0=VENT_LOCS['midpoint'][0],
    lat_0=VENT_LOCS['midpoint'][1],
    x_radius=350,
    y_radius=350,
    spacing=10,
    projected=True,
)
dem = produce_dem(
    grid,
    external_file=str(WORKING_DIR / 'data' / 'DEM_WGS84.tif'),
    plot_output=False,
)

#%% Make histogram

# Change registration
xe = np.hstack([grid.x.values, grid.x.values[-1] + grid.spacing]) - grid.spacing / 2
ye = np.hstack([grid.y.values, grid.y.values[-1] + grid.spacing]) - grid.spacing / 2

# Compute histogram and assign to DataArray
h, *_ = np.histogram2d(df.x, df.y, bins=[xe, ye])
h[h == 0] = np.nan
hist = grid.copy()
hist.data = h.T  # Because of NumPy array axis handling

# Break in half around vent midpoint to define two clusters
hist_A = hist.where(hist.y < y_0, drop=True)
hist_C = hist.where(hist.y > y_0, drop=True)

# Find maxima for each vent cluster
A_max = hist_A.where(hist_A == hist_A.max(), drop=True)
C_max = hist_C.where(hist_C == hist_C.max(), drop=True)

# Plot
if PLOT:
    fig, ax = plt.subplots()
    dem.plot.contour(ax=ax, levels=20, colors='black', linewidths=0.5)
    hist.plot.pcolormesh(ax=ax, cmap=cc.m_fire_r, cbar_kwargs=dict(label='# of events'))
    for loc in (A_max.x.values, A_max.y.values), (C_max.x.values, C_max.y.values):
        ax.add_artist(plt.Circle(loc, MAX_RADIUS, edgecolor='black', facecolor='none'))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{df.shape[0]} events in catalog, MAX_RADIUS = {MAX_RADIUS} m')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    # Tick formatting
    RADIUS = 300
    INC = 100
    for axis, ref_coord in zip([ax.xaxis, ax.yaxis], [x_0, y_0]):
        fmt = axis.get_major_formatter()
        fmt.set_useOffset(ref_coord)
        axis.set_major_formatter(fmt)
        plt.setp(axis.get_offset_text(), visible=False)
        axis.set_ticks(np.arange(-RADIUS, RADIUS + INC, INC) + ref_coord)

    fig.show()

#%% Label based upon proximity to histogram peak


def within_radius(true_loc, est_loc, radius):
    return np.linalg.norm(np.array(true_loc) - np.array(est_loc)) < radius


vent_locs = []
for x, y in zip(df.x, df.y):

    at_A = within_radius((A_max.x.values[0], A_max.y.values[0]), (x, y), MAX_RADIUS)
    at_C = within_radius((C_max.x.values[0], C_max.y.values[0]), (x, y), MAX_RADIUS)

    if at_A and not at_C:
        vent_locs.append('A')
    elif at_C and not at_A:
        vent_locs.append('C')
    else:  # Either not located or doubly-located
        vent_locs.append(None)

df['vent'] = vent_locs

# Remove rows with no location
df_locs = df[~pd.isnull(df.vent)]

#%% (OPTIONAL) Make area plot of labeled catalog

if PLOT:

    # Toggle plotting fraction of vent A vs. vent C in each window (otherwise plot totals)
    FRACTION = False

    # [s] Rolling window duration
    WINDOW = 60 * 60

    # Start and end on whole hours
    t_start = UTCDateTime('2016-07-27T05')
    t_end = UTCDateTime('2016-08-01T22')

    # Form array of UTCDateTimes
    t_vec = [t_start + t for t in np.arange(0, (t_end - t_start), WINDOW)]

    # In moving windows, get counts of vent A and vent C
    fraction_A = []
    fraction_C = []
    for t in t_vec:
        df_hr = df_locs[(df_locs.t >= t) & (df_locs.t < t + WINDOW)]
        vcounts = df_hr.vent.value_counts()
        if FRACTION:
            vcounts /= vcounts.sum()
        if hasattr(vcounts, 'A'):
            fraction_A.append(vcounts.A)
        else:
            fraction_A.append(0)
        if hasattr(vcounts, 'C'):
            fraction_C.append(vcounts.C)
        else:
            fraction_C.append(0)

    # Load in a single station's data and process (takes a while, can comment out for repeat
    # runs)
    tr = read(str(WORKING_DIR / 'data' / '3E_YIF1-5_50hz.pkl')).select(station='YIF3')[
        0
    ]
    tr.remove_response()
    tr.filter('bandpass', freqmin=0.2, freqmax=4, zerophase=True)

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(13, 5))

    # Subplot 1: Waveform
    axes[0].plot(tr.times('matplotlib'), tr.data, linewidth=0.5, color='black')
    axes[0].set_ylabel('Pressure (Pa)')

    # Subplot 2: Stacked area plot
    t_vec_mpl = [t.matplotlib_date for t in t_vec]
    axes[1].stackplot(
        t_vec_mpl,
        fraction_A,
        fraction_C,
        colors=(os.environ['VENT_A'], os.environ['VENT_C']),
        labels=('Vent A', 'Vent C'),
    )
    if FRACTION:
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    else:
        axes[1].set_ylabel('Number of labeled events')
    axes[1].autoscale(enable=True, axis='y', tight=True)

    # Overall x-axis formatting
    axes[-1].set_xlim(t_start.matplotlib_date, (t_end - WINDOW).matplotlib_date)
    loc = axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    # Add legend
    axes[-1].legend(loc='lower right')

    fig.show()

#%% Load in full dataset

print('Reading in full dataset...')
st_full = read(str(WORKING_DIR / 'data' / '3E_YIF1-5_50hz.pkl'))
print('Done')

WAVEFORM_DUR = 5  # [s] Duration of labeled waveform snippets

fs = st_full[0].stats.sampling_rate  # [Hz]

length_samples = int(WAVEFORM_DUR * fs)  # [samples]

print('Labeling waveforms...')
n = 0
st_label = Stream()
for i, row in df_locs.iterrows():

    st = st_full.copy().trim(row.t, row.t + WAVEFORM_DUR)
    for tr in st:
        tr.stats.vent = row.vent
        tr.stats.event_info = dict(utm_x=row.x, utm_y=row.y, origin_time=row.t)
        tr.data = tr.data[:length_samples]

    st_label += st

    if (i + 1) % 10 == 0:
        st_label.write(
            str(WORKING_DIR / 'data' / 'labeled' / f'label_{n:03}.pkl'), format='PICKLE'
        )
        st_label = Stream()
        print(f'{(i / df_locs.shape[0]) * 100:.1f}%')  # TODO: Goes wayy over 100%, lol
        n += 1

# Handle last one
st_label.write(
    str(WORKING_DIR / 'data' / 'labeled' / f'label_{n:03}.pkl'), format='PICKLE'
)

print('Done')

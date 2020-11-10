"""
Read in CSV files corresponding to a catalog, define the peaks of the histograms around
each vent, and label events based upon proximity to these peaks.
"""

import json
from pathlib import Path

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm
from obspy import Stream, UTCDateTime, read
from rtm import define_grid, produce_dem

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Define which catalog to label
catalog_dir = WORKING_DIR / 'label' / 'catalogs' / 'height_4_spacing_30_agc_60'

# [m] Maximum distance from vent location estimate to a given event
MAX_RADIUS = 40

# Read in entire catalog to pandas DataFrame
df = pd.DataFrame()
for csv in sorted(catalog_dir.glob('catalog_??.csv')):
    df = pd.concat([df, pd.read_csv(csv)], ignore_index=True)
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
    grid, external_file=str(WORKING_DIR / 'data' / 'DEM_WGS84.tif'), plot_output=False,
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
PLOT = True
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

#%% Load in full dataset

print('Reading in full dataset...')
st_full = read('data/3E_YIF1-5_50hz.pkl').interpolate(
    sampling_rate=20, method='lanczos', a=20
)
print('Done')

WAVEFORM_DUR = 10  # [s] Duration of labeled waveform snippets

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
        print(f'{(i / df_locs.shape[0]) * 100:.1f}%')
        n += 1

# Handle last one
st_label.write(
    str(WORKING_DIR / 'data' / 'labeled' / f'label_{n:03}.pkl'), format='PICKLE'
)

print('Done')

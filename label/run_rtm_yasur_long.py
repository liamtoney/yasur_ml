#%% Import packages and load full dataset

# isort: skip_file

import json

import numpy as np
import utm
from obspy import UTCDateTime, read
from rtm import (
    define_grid,
    get_peak_coordinates,
    grid_search,
    plot_time_slice,
    process_waveforms,
    produce_dem,
)

# Load in full dataset
st_full = read('data/3E_YIF1-5_50hz.pkl')

#%% Set parameters

# Info here: https://portal.opentopography.org/dataspace/dataset?opentopoID=OTDS.072019.4326.1
EXTERNAL_FILE = '/Users/ldtoney/work/yasur_ml/data/DEM_WGS84.tif'

# Load vent locs
with open('/Users/ldtoney/work/yasur_ml/yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

FREQ_MIN = 0.2  # [Hz] Lower bandpass corner
FREQ_MAX = 4  # [Hz] Upper bandpass corner

DECIMATION_RATE = 20  # [Hz] New sampling rate to use for decimation
AGC_PARAMS = dict(win_sec=120, method='walker')

# Data window
STARTTIME = UTCDateTime('2016-07-29T02:00')
ENDTIME = STARTTIME + 30 * 60

# Grid params
GRID_SPACING = 10  # [m]
SEARCH_LON_0 = VENT_LOCS['midpoint'][0]
SEARCH_LAT_0 = VENT_LOCS['midpoint'][1]
SEARCH_X = 350
SEARCH_Y = 350

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'
FILENAME_ROOT = 'yasur_rtm_DH2_no_YIF6'  # output filename root prefix
FDTD_DIR = '/Users/ldtoney/work/yasur_ml/label/fdtd/'  # Where travel time lookup table is located

XY_GRID = 350
CONT_INT = 5
ANNOT_INT = 50

#%% Create grids

grid = define_grid(
    lon_0=SEARCH_LON_0,
    lat_0=SEARCH_LAT_0,
    x_radius=SEARCH_X,
    y_radius=SEARCH_Y,
    spacing=GRID_SPACING,
    projected=True,
)

dem = produce_dem(grid, external_file=EXTERNAL_FILE, plot_output=False)

#%% read in data

# Trim to what we want
st = st_full.copy().trim(starttime=STARTTIME, endtime=ENDTIME)

st_proc = process_waveforms(
    st,
    freqmin=FREQ_MIN,
    freqmax=FREQ_MAX,
    envelope=True,
    decimation_rate=DECIMATION_RATE,
    agc_params=None,
    normalize=True,
)

#%% process grid

S = grid_search(
    processed_st=st_proc,
    grid=grid,
    time_method='fdtd',
    stack_method='sum',
    FILENAME_ROOT=FILENAME_ROOT,
    FDTD_DIR=FDTD_DIR,
)

#%% Plot maximum slice

fig_slice = plot_time_slice(
    S,
    st_proc,
    time_slice=None,
    label_stations=True,
    dem=dem,
    plot_peak=True,
    xy_grid=XY_GRID,
    cont_int=CONT_INT,
    annot_int=ANNOT_INT,
)
fig_slice.axes[1].set_ylim(top=st_proc.count())

#%% Automatically determine vent locations (DRAFT)

import matplotlib.pyplot as plt

MAX_RADIUS = 50  # [m] Radius of circle around each vent

time_max, y_max, x_max, *_ = get_peak_coordinates(
    S, global_max=False, height=3, min_time=10, unproject=False
)

fig, ax = plt.subplots()

dem.plot.contour(ax=ax, levels=20, colors='black', linewidths=0.5)
ax.set_aspect('equal')

# Tick formatting
RADIUS = 300
INC = 100
x_0, y_0, *_ = utm.from_latlon(SEARCH_LAT_0, SEARCH_LON_0)
for axis, ref_coord in zip([ax.xaxis, ax.yaxis], [x_0, y_0]):
    fmt = axis.get_major_formatter()
    fmt.set_useOffset(ref_coord)
    axis.set_major_formatter(fmt)
    plt.setp(axis.get_offset_text(), visible=False)
    axis.set_ticks(np.arange(-RADIUS, RADIUS + INC, INC) + ref_coord)

# Convert vent locs to UTM
vent_locs_utm = {}
for vent, loc in {k: VENT_LOCS[k] for k in VENT_LOCS if k != 'midpoint'}.items():
    utm_x, utm_y, *_ = utm.from_latlon(*loc[::-1])
    vent_locs_utm[vent] = [utm_x, utm_y]

# Plot vents and associated radii thresholds
for vent, loc in vent_locs_utm.items():
    ax.scatter(*loc, marker='^', color='green', edgecolors='black')
    ax.add_artist(plt.Circle(loc, MAX_RADIUS, color='lightgrey', zorder=-100))


def within_radius(true_loc, est_loc, radius):
    return np.linalg.norm(np.array(true_loc) - np.array(est_loc)) < radius


def color_code(vent_loc):
    if vent_loc == 'A':
        color = 'blue'
    elif vent_loc == 'C':
        color = 'red'
    else:  # vent is None
        color = 'black'
    return color


vent_locs = []
for x, y in zip(x_max, y_max):

    at_A = within_radius(vent_locs_utm['A'], (x, y), MAX_RADIUS)
    at_C = within_radius(vent_locs_utm['C'], (x, y), MAX_RADIUS)

    if at_A and not at_C:
        vent_locs.append('A')
    elif at_C and not at_A:
        vent_locs.append('C')
    else:  # Either not located or doubly-located
        vent_locs.append(None)

print(vent_locs)

for i, (x, y, vent) in enumerate(zip(x_max, y_max, vent_locs)):
    ax.scatter(x, y, edgecolors='black', facecolors=color_code(vent))

ax.set_title(f'MAX_RADIUS = {MAX_RADIUS} m')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')

fig.show()

#%% Window these times and label using manually defined vents

import matplotlib.pyplot as plt

DUR = 30  # [s] Time window for signal, probably should be less than the min_time above

fig = plt.figure(figsize=(12, 8))
stp = st.copy().remove_response()
stp.filter('bandpass', freqmin=FREQ_MIN, freqmax=FREQ_MAX)
stp.taper(0.05)
stp.plot(fig=fig, equal_scale=True, method='full')

for ax in fig.axes:
    for t, vent in zip(time_max, vent_locs):
        ax.axvspan(
            t.matplotlib_date,
            (t + DUR).matplotlib_date,
            alpha=0.2,
            color=color_code(vent),
            linewidth=0,
        )
fig.show()

#%% Export CSV

import pandas as pd

df = pd.DataFrame(dict(x=x_max, y=y_max, t=time_max, vent=vent_locs))
df.to_csv('label/catalog.csv', index=False)

from obspy import UTCDateTime
from rtm import (
    calculate_time_buffer,
    define_grid,
    get_peak_coordinates,
    grid_search,
    plot_time_slice,
    process_waveforms,
    produce_dem,
)
from waveform_collection import gather_waveforms

# curl -O https://cloud.sdsc.edu/v1/AUTH_opentopography/hosted_data/OTDS.072019.4326.1/raster/DEM_WGS84.tif
EXTERNAL_FILE = '/Users/ldtoney/work/yasur_ml/label/DEM_WGS84.tif'

FREQ_MIN = 0.2  # [Hz] Lower bandpass corner
FREQ_MAX = 4  # [Hz] Upper bandpass corner

DECIMATION_RATE = 20  # [Hz] New sampling rate to use for decimation
DH = 10
AGC_PARAMS = dict(win_sec=120, method='walker')

# data params
STARTTIME = UTCDateTime("2016-07-29T03:00")
ENDTIME = STARTTIME + 10 * 60

SOURCE = 'IRIS'  # local, IRIS
NETWORK = '3E'  # YS, 3E
STATION = 'YIF?'
LOCATION = '*'
CHANNEL = '*'

# grid params
# vent mid
SEARCH_LON_0 = 169.44804400000001
SEARCH_LAT_0 = -19.528539500000001

SEARCH_X = 350
SEARCH_Y = 350
DH_SEARCH = DH

MAX_STATION_DIST = 0.8  # [km] Max. dist. from grid center to station (approx.)

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'
FILENAME_ROOT = 'yasur_rtm_DH2'  # output filename root prefix
FDTD_DIR = '/Users/ldtoney/work/yasur_ml/label/fdtd/'  # Where travel time lookup table is located

XY_GRID = 350
CONT_INT = 5
ANNOT_INT = 50

#%% Create grids

search_grid = define_grid(
    lon_0=SEARCH_LON_0,
    lat_0=SEARCH_LAT_0,
    x_radius=SEARCH_X,
    y_radius=SEARCH_Y,
    spacing=DH_SEARCH,
    projected=True,
)

search_dem = produce_dem(search_grid, external_file=EXTERNAL_FILE, plot_output=True)


#%% read in data

# Automatically determine appropriate time buffer in s
time_buffer = calculate_time_buffer(search_grid, MAX_STATION_DIST)

st = gather_waveforms(
    source=SOURCE,
    network=NETWORK,
    station=STATION,
    location=LOCATION,
    channel=CHANNEL,
    starttime=STARTTIME,
    endtime=ENDTIME,
    time_buffer=time_buffer,
)

st_proc = process_waveforms(
    st,
    freqmin=FREQ_MIN,
    freqmax=FREQ_MAX,
    envelope=True,
    decimation_rate=DECIMATION_RATE,
    agc_params=AGC_PARAMS,
    normalize=True,
)

#%% process grid

S = grid_search(
    processed_st=st_proc,
    grid=search_grid,
    time_method='fdtd',
    starttime=STARTTIME,
    endtime=ENDTIME,
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
    dem=search_dem,
    plot_peak=True,
    xy_grid=XY_GRID,
    cont_int=CONT_INT,
    annot_int=ANNOT_INT,
)

#%% Plot the maxes identified to figure out which vent

import matplotlib.pyplot as plt

time_max, y_max, x_max, *_ = get_peak_coordinates(
    S, global_max=False, height=4.5, min_time=30, unproject=False
)

fig, ax = plt.subplots()

search_dem.plot.contour(ax=ax, levels=20, colors='black', linewidths=0.5)
ax.set_aspect('equal')

for i, (xmax, ymax, tmax) in enumerate(zip(x_max, y_max, time_max)):
    ax.scatter(xmax, ymax, edgecolors='black', facecolors='orange')
    ax.text(xmax, ymax, f'{i}')  # Label with index

# Manually determined vent locations
vents = ['A', 'A', 'A', 'C']

#%% Window these times and label using manually defined vents

import matplotlib.pyplot as plt

DUR = 30  # [s] Time window for signal

fig = plt.figure(figsize=(12, 8))
stp = st.copy().remove_response()
stp.filter('bandpass', freqmin=FREQ_MIN, freqmax=FREQ_MAX)
stp.taper(0.05)
stp.plot(fig=fig, equal_scale=False)

for ax in fig.axes:
    for tmax, vent in zip(time_max, vents):
        if vent == 'A':
            color = 'red'
        else:  # vent == 'C'
            color = 'green'
        ax.axvspan(
            tmax.matplotlib_date,
            (tmax + DUR).matplotlib_date,
            alpha=0.2,
            color=color,
            linewidth=0,
        )
fig.show()

#%% Import packages and load full dataset

# isort: skip_file

import json
from datetime import datetime

import pandas as pd
from obspy import read
from rtm import (
    define_grid,
    get_peak_coordinates,
    grid_search,
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

# Detection params
MAX_RADIUS = 30  # [m] Radius of circle around each vent
HEIGHT_THRESHOLD = 4  # Minimum stack function value required
MIN_TIME_SPACING = 30  # [s] Minimum time between adjacent peaks

# Bulk run params
CHUNK_DURATION = 3  # [h] Length of data chunk to run RTM on

# Grid params
GRID_SPACING = 10  # [m]
SEARCH_LON_0 = VENT_LOCS['midpoint'][0]
SEARCH_LAT_0 = VENT_LOCS['midpoint'][1]
SEARCH_X = 350
SEARCH_Y = 350

# FDTD grid info
FILENAME_ROOT = 'yasur_rtm_DH2_no_YIF6'  # output filename root prefix
FDTD_DIR = '/Users/ldtoney/work/yasur_ml/label/fdtd/'  # Where travel time lookup table is located

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

#%% Run in chunks

full_starttime = st_full[0].stats.starttime
full_endtime = st_full[0].stats.endtime

chunk_duration_sec = CHUNK_DURATION * 60 * 60  # [s]

# Initialize stuff
num_hours_processed = 0  # [h]
starttime = full_starttime
n = 0

t0 = datetime.now()
while True:

    # START-OF-LOOP STUFF
    endtime = starttime + chunk_duration_sec
    leave = False
    if endtime > full_endtime:
        endtime = full_endtime
        leave = True

    # Trim to this chunk's window
    st = st_full.copy().trim(starttime=starttime, endtime=endtime)

    # Process waveforms
    st_proc = process_waveforms(
        st,
        freqmin=FREQ_MIN,
        freqmax=FREQ_MAX,
        taper_length=30,
        envelope=True,
        decimation_rate=DECIMATION_RATE,
        agc_params=dict(win_sec=120, method='walker'),
        normalize=True,
    )

    # Grid search
    S = grid_search(
        processed_st=st_proc,
        grid=grid,
        time_method='fdtd',
        stack_method='sum',
        FILENAME_ROOT=FILENAME_ROOT,
        FDTD_DIR=FDTD_DIR,
    )

    # Automatically determine vent locations
    time_max, y_max, x_max, *_ = get_peak_coordinates(
        S,
        global_max=False,
        height=HEIGHT_THRESHOLD,
        min_time=MIN_TIME_SPACING,
        unproject=False,
    )

    # Export to CSV
    df = pd.DataFrame(dict(x=x_max, y=y_max, t=time_max))
    df.to_csv(f'label/catalog/catalog_{n:02}.csv', index=False)

    # END-OF-LOOP STUFF
    if leave:
        break
    num_hours_processed += CHUNK_DURATION
    print(f'{num_hours_processed:g} hrs processed')
    starttime += chunk_duration_sec
    n += 1

print(datetime.now() - t0)

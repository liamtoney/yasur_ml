"""
Run RTM on chunks of data to assemble a catalog of (x, y, t) stored as a CSV file. x and
y are in meters in UTM zone 59S. t is in UTC.
"""

# isort: skip_file

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from obspy import read
from rtm import (
    define_grid,
    get_peak_coordinates,
    grid_search,
    process_waveforms,
    produce_dem,
)

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load in full dataset
st_full = read(str(WORKING_DIR / 'data' / '3E_YIF1-5_50hz.pkl'))

#%% Set parameters

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# [Hz] Bandpass corners (David's local RTM paper uses 0.2–4 Hz)
FREQ_MIN = 0.2
FREQ_MAX = 4

DECIMATION_RATE = 20  # [Hz] New sampling rate to use for decimation

# [s] AGC window (David's local RTM paper uses 120 s)
AGC_WINDOW = 60

# Detection params
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
FDTD_DIR = WORKING_DIR / 'label' / 'fdtd'  # Where travel time lookup table is located

# Create catalog CSV name
catalog_csv = (
    WORKING_DIR
    / 'label'
    / 'catalogs'
    / f'height_{HEIGHT_THRESHOLD}_spacing_{MIN_TIME_SPACING}_agc_{AGC_WINDOW}.csv'
)

#%% Create grids

grid = define_grid(
    lon_0=SEARCH_LON_0,
    lat_0=SEARCH_LAT_0,
    x_radius=SEARCH_X,
    y_radius=SEARCH_Y,
    spacing=GRID_SPACING,
    projected=True,
)

# DEM info here: https://portal.opentopography.org/dataspace/dataset?opentopoID=OTDS.072019.4326.1
# Can obtain with curl -O https://cloud.sdsc.edu/v1/AUTH_opentopography/hosted_data/OTDS.072019.4326.1/raster/DEM_WGS84.tif
dem = produce_dem(
    grid, external_file=str(WORKING_DIR / 'data' / 'DEM_WGS84.tif'), plot_output=False
)

#%% Run in chunks

full_starttime = st_full[0].stats.starttime
full_endtime = st_full[0].stats.endtime

chunk_duration_sec = CHUNK_DURATION * 60 * 60  # [s]

# Initialize stuff
num_hours_processed = 0  # [h]
starttime = full_starttime
df_total = pd.DataFrame()

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
        taper_length=30,  # [s]
        envelope=True,
        decimation_rate=DECIMATION_RATE,
        agc_params=dict(win_sec=AGC_WINDOW, method='walker'),
        normalize=True,
    )

    # Grid search
    S = grid_search(
        processed_st=st_proc,
        grid=grid,
        time_method='fdtd',
        stack_method='sum',
        FILENAME_ROOT=FILENAME_ROOT,
        FDTD_DIR=str(FDTD_DIR) + '/',  # Hacky :(
    )

    # Automatically determine vent locations
    time_max, y_max, x_max, *_ = get_peak_coordinates(
        S,
        global_max=False,
        height=HEIGHT_THRESHOLD,
        min_time=MIN_TIME_SPACING,
        unproject=False,
    )

    # Create DataFrame and append to main DataFrame
    df = pd.DataFrame(dict(x=x_max, y=y_max, t=time_max))
    df_total = pd.concat([df_total, df], ignore_index=True)

    # END-OF-LOOP STUFF
    if leave:
        break
    num_hours_processed += CHUNK_DURATION
    print(f'{num_hours_processed:g} hrs processed')
    starttime += chunk_duration_sec

# Write entire catalog to CSV
df_total.to_csv(catalog_csv, index=False)

# Display total run time
print('Elapsed time:')
print(datetime.now() - t0)

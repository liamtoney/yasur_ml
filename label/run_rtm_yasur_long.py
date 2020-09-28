from obspy import UTCDateTime
from waveform_collection import gather_waveforms
from rtm import define_grid, produce_dem, process_waveforms, calculate_time_buffer, grid_search
from rtm import (plot_time_slice, get_peak_coordinates)

SVDIR = '/Users/dfee/Documents/rtm/figs/'
EXTERNAL_FILE = '/Users/dfee/Documents/vanuatu/DEM/DEM_Union_UAV_161116_sm101.tif'
DATADIR='/Users/dfee/Documents/vanuatu/data/converted/'

FREQ_MIN = .2          # [Hz] Lower bandpass corner
FREQ_MAX = 4            # [Hz] Upper bandpass corner

DECIMATION_RATE = 40  # [Hz] New sampling rate to use for decimation
SMOOTH_WIN = None        # [s] Smoothing window duration
DH = 4
AGC_PARAMS = dict(win_sec=120, method='walker')

#data params
STARTTIME = UTCDateTime("2016-07-29T03:00")
ENDTIME = STARTTIME + 10*60

SOURCE = 'IRIS' #local, IRIS
NETWORK = '3E' #YS, 3E
STATION = 'YIF?'
LOCATION = '*'
CHANNEL = '*'

# grid params
#vent mid
SEARCH_LON_0 = 169.44804400000001
SEARCH_LAT_0 = -19.528539500000001

SEARCH_X = 350
SEARCH_Y = 350
DH_SEARCH = DH

MAX_STATION_DIST = 0.8  # [km] Max. dist. from grid center to station (approx.)
CEL = 343.48

STACK_METHOD = 'sum'  # Choose either 'sum' or 'product'
TIME_METHOD1 = 'celerity'  # Choose either 'celerity' or 'fdtd'
TIME_METHOD2 = 'fdtd'  # Choose either 'celerity' or 'fdtd'
FILENAME_ROOT = 'yasur_rtm_DH2'    #output filename root prefix
FDTD_DIR='/Users/dfee/code//infraFDTD_new/'+FILENAME_ROOT+'/'    # directory for FDTD input files

XY_GRID = 350
CONT_INT = 5
ANNOT_INT = 50


#%% Create grids

search_grid = define_grid(lon_0=SEARCH_LON_0, lat_0=SEARCH_LAT_0,
                          x_radius=SEARCH_X, y_radius=SEARCH_Y,
                          spacing=DH_SEARCH, projected=True)

search_dem = produce_dem(search_grid, external_file=EXTERNAL_FILE, plot_output=True)


#%% read in data

# Automatically determine appropriate time buffer in s
time_buffer = calculate_time_buffer(search_grid, MAX_STATION_DIST)

st = gather_waveforms(source=SOURCE, network=NETWORK, station=STATION,
                   location=LOCATION, channel=CHANNEL, starttime=STARTTIME,
                   endtime=ENDTIME, time_buffer=time_buffer)

st_proc_semb = process_waveforms(st, freqmin=FREQ_MIN, freqmax=FREQ_MAX,
                            envelope=False, smooth_win=SMOOTH_WIN,
                            decimation_rate=DECIMATION_RATE,
                            agc_params=AGC_PARAMS,
                            normalize=True, plot_steps=False)

#%% process grid

time_kwargs_fdtd = {"FILENAME_ROOT" : FILENAME_ROOT, "FDTD_DIR": FDTD_DIR}
S_fdtd_semb = grid_search(processed_st=st_proc_semb, grid=search_grid,
                                  time_method=TIME_METHOD2,
                                  starttime=STARTTIME, endtime=ENDTIME,
                                  window=5, overlap=.5,
                                  stack_method='semblance',
                                  **time_kwargs_fdtd)

#%% plot

fig_slice_semb = plot_time_slice(S_fdtd_semb, st_proc_semb, time_slice=None,
                                 label_stations=True, dem=search_dem,
                                 plot_peak=True, xy_grid=XY_GRID, cont_int=CONT_INT,
                                 annot_int=ANNOT_INT)

time_max_semb, y_max_semb, x_max_semb, _, _ = get_peak_coordinates(S_fdtd_semb,
                                                                   global_max=True,
                                                                   height=.6,
                                                                   min_time=2,
                                                                   unproject=False)

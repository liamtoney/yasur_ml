#!/usr/bin/env python

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
import utm
import xarray
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from rtm import define_grid

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# Region is from running GMT with -RVU for Vanuatu
PAD = 0.14
VANUATU_REGION = (166.525 - PAD, 170.235 + PAD, -20.2489 - PAD, -13.0734 + PAD)

# Gather station coordinates
STATION = 'YIF1,YIF2,YIF3,YIF4,YIF5,YIF6'
net = Client('IRIS').get_stations(
    network='3E',
    station=STATION,
    starttime=UTCDateTime(2016, 1, 1),
    endtime=UTCDateTime(2016, 12, 31),
    level='station',
)[0]
sta_lon = [sta.longitude for sta in net]
sta_lat = [sta.latitude for sta in net]
sta_code = [sta.code for sta in net]

# Convert vent midpoint to UTM
x_0, y_0, *_ = utm.from_latlon(VENT_LOCS['midpoint'][1], VENT_LOCS['midpoint'][0])

# Radius around vent midpoint to use for region [m]
RADIUS = 800

# RTM grid search radius [m] (NEED TO UPDATE IF build_catalog.py CHANGES!)
RTM_RADIUS = 350

# Read in and process DEM to use, command to create was:
# gdalwarp -t_srs EPSG:32759 -r cubicspline DEM_Union_UAV_161116_sm101.tif DEM_Union_UAV_161116_sm101_UTM.tif
DEM = WORKING_DIR / 'data' / 'DEM_Union_UAV_161116_sm101_UTM.tif'
dem = xarray.open_rasterio(DEM).squeeze()
dem = dem.assign_coords(x=(dem.x.data - x_0))
dem = dem.assign_coords(y=(dem.y.data - y_0))
dem = dem.where(dem > dem.nodatavals[0])
dem = dem.where(
    (dem.x >= -RADIUS) & (dem.x <= RADIUS) & (dem.y >= -RADIUS) & (dem.y <= RADIUS),
    drop=True,
)


# Define transform from (lat, lon) to (x, y) from vent midpoint
def transform(longitude, latitude):
    lons = np.atleast_1d(longitude)
    lats = np.atleast_1d(latitude)
    x = []
    y = []
    for lon, lat in zip(lons, lats):
        utm_x, utm_y, *_ = utm.from_latlon(lat, lon)
        x.append(utm_x - x_0)
        y.append(utm_y - y_0)
    return np.array(x), np.array(y)


pygmt.config(FONT='14p')

fig = pygmt.Figure()

# (a) Location of Yasur
fig.coast(
    region=VANUATU_REGION,
    projection='T{}/{}/3i'.format(
        np.mean(VANUATU_REGION[:2]), np.mean(VANUATU_REGION[2:])
    ),
    land='lightgrey',
    water='lightblue',
    shorelines=True,
    frame='af',
    resolution='f',
)
fig.plot(*VENT_LOCS['midpoint'], style='t0.4c', color='red', pen=True)  # Center of (b)
fig.shift_origin(xshift='1.6i', yshift='4.1i')  # For globe inset
fig.coast(
    region='g',
    projection='G{}/{}/2i'.format(
        np.mean(VANUATU_REGION[:2]), np.mean(VANUATU_REGION[2:])
    ),
    land='lightgrey',
    water='lightblue',
    shorelines=True,
    frame='g',
    area_thresh='500/0/1',
    resolution='i',
)
verts = [
    (VANUATU_REGION[0], VANUATU_REGION[2]),
    (VANUATU_REGION[0], VANUATU_REGION[3]),
    (VANUATU_REGION[1], VANUATU_REGION[3]),
    (VANUATU_REGION[1], VANUATU_REGION[2]),
    (VANUATU_REGION[0], VANUATU_REGION[2]),
]
fig.plot(data=np.array(verts), straight_line='p', pen='1p,red')

# (b) Station map
fig.shift_origin(xshift='3.5i', yshift='-4.11i')

HEIGHT = 5.7  # [in]
fig.grdimage(
    dem,
    region=(-RADIUS, RADIUS, -RADIUS, RADIUS),
    cmap='white',
    shading=True,
    projection=f'X{HEIGHT}i',
    frame=['a500f100', 'x+l"Easting (m)"', 'y+l"Northing (m)"', 'WSen'],
)
fig.grdcontour(dem, interval=10, annotation='100+u" m"')

# Read in entire catalog to pandas DataFrame
catalog_csv = WORKING_DIR / 'label' / 'catalogs' / 'height_4_spacing_30_agc_60.csv'
df = pd.read_csv(catalog_csv)

# Define grid (TODO: Needs to be the same as the original RTM grid!)
grid = define_grid(
    lon_0=VENT_LOCS['midpoint'][0],
    lat_0=VENT_LOCS['midpoint'][1],
    x_radius=350,
    y_radius=350,
    spacing=10,
    projected=True,
)

# Make histogram
xe = np.hstack([grid.x.values, grid.x.values[-1] + grid.spacing]) - grid.spacing / 2
ye = np.hstack([grid.y.values, grid.y.values[-1] + grid.spacing]) - grid.spacing / 2
h, *_ = np.histogram2d(df.x, df.y, bins=[xe, ye])
h[h == 0] = np.nan
hist = grid.copy()
hist.data = h.T  # Because of NumPy array axis handling
hist = hist.assign_coords(x=(hist.x.data - x_0))
hist = hist.assign_coords(y=(hist.y.data - y_0))

# Plot histogram and add colorbar
pygmt.makecpt(
    cmap='inferno', reverse=True, series=[hist.min().values, hist.max().values]
)
fig.grdview(hist, cmap=True, T='+s')
fig.colorbar(position=f'JMR+w{HEIGHT}i', frame=f'a100f50+l"# of locations"')

verts = [
    (-RTM_RADIUS, -RTM_RADIUS),
    (-RTM_RADIUS, RTM_RADIUS),
    (RTM_RADIUS, RTM_RADIUS),
    (RTM_RADIUS, -RTM_RADIUS),
    (-RTM_RADIUS, -RTM_RADIUS),
]
fig.plot(data=np.array(verts), straight_line='p', pen='0.75p,black,-')

# Plot ellipses used for event labeling
vent_pen = '2p'
for vent in 'A', 'C':

    # Ellipses
    fig.plot(
        data=str(WORKING_DIR / 'plot_scripts' / f'{vent}_ellipse.xy'),
        pen=vent_pen + ',' + os.environ[f'VENT_{vent}'],
    )

    # Dummy for legend
    fig.plot(
        x=-4747,
        y=-4747,
        style='c0.3c',
        pen=vent_pen + ',' + os.environ[f'VENT_{vent}'],
        label=f'"Vent {vent}"',
    )

fig.plot(
    *transform(sta_lon, sta_lat),
    style='i0.4c',
    pen='0.75p',
    color='mediumseagreen',
    label='Station',
)

fig.legend()

fig.show(method='external')

# fig.savefig('/Users/ldtoney/Downloads/station_map.png', dpi=400)

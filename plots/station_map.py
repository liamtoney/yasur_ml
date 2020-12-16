import json
import os
from pathlib import Path

import numpy as np
import pygmt
import utm
import xarray
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

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
# gdalwarp -t_srs EPSG:32759 DEM_Union_UAV_161116_sm101.tif DEM_Union_UAV_161116_sm101_UTM.tif
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
fig.shift_origin(xshift='1.6i', yshift='4i')  # For globe inset
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
fig.shift_origin(xshift='3.5i', yshift='-4.01i')
fig.grdimage(
    dem,
    region=(-RADIUS, RADIUS, -RADIUS, RADIUS),
    cmap='gray',
    projection='X5.7i',
    frame=['a500f100', 'x+l"Easting (m)"', 'y+l"Northing (m)"', 'WSen'],
)
fig.grdimage(
    dem, shading=True, cmap='white', t=70,
)
fig.grdcontour(dem, interval=10, annotation='100+u" m"')
verts = [
    (-RTM_RADIUS, -RTM_RADIUS),
    (-RTM_RADIUS, RTM_RADIUS),
    (RTM_RADIUS, RTM_RADIUS),
    (RTM_RADIUS, -RTM_RADIUS),
    (-RTM_RADIUS, -RTM_RADIUS),
]
fig.plot(data=np.array(verts), straight_line='p', pen='0.75p,black,-')
vent_style = 'c0.3c'
vent_pen = '0.75p'
fig.plot(
    *transform(*VENT_LOCS['A']),
    style=vent_style,
    pen=vent_pen,
    color=os.environ['VENT_A'],
    label='"Vent A"',
)
fig.plot(
    *transform(*VENT_LOCS['C']),
    style=vent_style,
    pen=vent_pen,
    color=os.environ['VENT_C'],
    label='"Vent C"',
)
fig.plot(
    *transform(sta_lon, sta_lat),
    style='i0.4c',
    pen=vent_pen,
    color='mediumseagreen',
    label='Station',
)
fig.text(
    x=transform(sta_lon, sta_lat)[0],
    y=transform(sta_lon, sta_lat)[1],
    text=sta_code,
    font='white=~1p',
    justify='LM',
    D='0.13i/-0.01i',
)
fig.legend()

# Plot (a) and (b) tags (hacky)
tag_kwargs = dict(y=RADIUS, no_clip=True, justify='TL', font='18p')
fig.text(x=-RADIUS - 1625, text='(a)', **tag_kwargs)
fig.text(x=-RADIUS - 200, text='(b)', **tag_kwargs)

fig.show(method='external')

# fig.savefig(WORKING_DIR / 'figures' / 'station_map.png', dpi=400)

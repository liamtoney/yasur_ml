import json
import os
from pathlib import Path

import numpy as np
import pygmt
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# DEM file to use
DEM = WORKING_DIR / 'data' / 'DEM_Union_UAV_161116_sm101.tif'

# From GVP (https://volcano.si.edu/volcano.cfm?vn=257100)
YASUR_COORDS = (169.447, -19.532)

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

# Main map extent
RADIUS = 0.008  # deg
MAIN_REGION = (
    VENT_LOCS['midpoint'][0] - RADIUS,
    VENT_LOCS['midpoint'][0] + RADIUS,
    VENT_LOCS['midpoint'][1] - RADIUS,
    VENT_LOCS['midpoint'][1] + RADIUS,
)

pygmt.config(FORMAT_GEO_MAP='D')

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
fig.plot(*YASUR_COORDS, style='t0.3c', color='red', pen=True)
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
fig.shift_origin(xshift='3.5i', yshift='-3.8i')
fig.grdimage(
    str(DEM), cmap='gray30,white', region=MAIN_REGION, projection='M5i', frame='a0.005',
)
fig.grdcontour(str(DEM), interval=10, annotation='100+u" m"')
vent_style = 'c0.3c'
vent_pen = '0.75p'
fig.plot(
    *VENT_LOCS['A'],
    style=vent_style,
    pen=vent_pen,
    color=os.environ['VENT_A'],
    label='"Vent A"',
)
fig.plot(
    *VENT_LOCS['C'],
    style=vent_style,
    pen=vent_pen,
    color=os.environ['VENT_C'],
    label='"Vent C"',
)
fig.plot(
    sta_lon, sta_lat, style='i0.4c', pen=vent_pen, color='red', label='Station',
)
fig.text(
    x=sta_lon, y=sta_lat, text=sta_code, font='10p,white=~1p', justify='LM', D='0.1i/0'
)
scale_loc = VENT_LOCS['midpoint']
fig.basemap(map_scale='jBR+c{0}/{1}+w200e+o0.3i/0.4i'.format(*scale_loc))
fig.legend()

fig.show(method='external')

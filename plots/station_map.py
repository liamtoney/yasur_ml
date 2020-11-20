import json
import os
from pathlib import Path

import numpy as np
import pygmt

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# From GVP (https://volcano.si.edu/volcano.cfm?vn=257100)
YASUR_COORDS = (169.447, -19.532)

# Region is from running GMT with -RVU for Vanuatu
PAD = 0.14
VANUATU_REGION = (166.525 - PAD, 170.235 + PAD, -20.2489 - PAD, -13.0734 + PAD)

# Main map extent
RADIUS = 0.01  # deg
MAIN_REGION = (
    VENT_LOCS['midpoint'][0] - RADIUS,
    VENT_LOCS['midpoint'][0] + RADIUS,
    VENT_LOCS['midpoint'][1] - RADIUS,
    VENT_LOCS['midpoint'][1] + RADIUS,
)

pygmt.config(FORMAT_GEO_MAP='D')

fig = pygmt.Figure()

# Main map
fig.grdimage(
    '@earth_relief_01s',
    region=MAIN_REGION,
    projection='M6i',
    shading=True,
    frame='a0.005',
)
fig.plot(
    *VENT_LOCS['A'],
    style='c0.4c',
    pen=True,
    color=os.environ['VENT_A'],
    label='"Vent A"'
)
fig.plot(
    *VENT_LOCS['C'],
    style='c0.4c',
    pen=True,
    color=os.environ['VENT_C'],
    label='"Vent C"'
)
scale_loc = VENT_LOCS['midpoint']
fig.basemap(map_scale='jBR+c{0}/{1}+w200e+o0.3i/0.4i'.format(*scale_loc))
fig.legend()

# Vanuatu inset
fig.shift_origin(xshift='7.5i', yshift='0.5i')
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

# Globe inset
fig.shift_origin(xshift='1.6i', yshift='4i')
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

fig.show(method='external')

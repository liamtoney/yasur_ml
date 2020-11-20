import numpy as np
import pygmt

# From GVP (https://volcano.si.edu/volcano.cfm?vn=257100)
YASUR_COORDS = (169.447, -19.532)

# Region is from running GMT with -RVU for Vanuatu
PAD = 0.14
VANUATU_REGION = (166.525 - PAD, 170.235 + PAD, -20.2489 - PAD, -13.0734 + PAD)

# Main map extent
MAIN_REGION = (169.4, 169.5, -19.6, -19.5)

pygmt.config(FORMAT_GEO_MAP='D')

fig = pygmt.Figure()

# Main map
fig.grdimage(
    '@earth_relief_01s',
    region=MAIN_REGION,
    projection='M6i',
    shading=True,
    frame='a0.02f0.01',
)
# fig.plot(*YASUR_COORDS, style='t0.3c', color='red', pen=True)
scale_loc = (169.485, -19.594)
fig.basemap(map_scale='g{0}/{1}+c{0}/{1}+w2+f+lkm'.format(*scale_loc))

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

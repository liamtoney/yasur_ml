#!/usr/bin/env python

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import utm
import xarray as xr
from matplotlib.ticker import MultipleLocator
from obspy import read

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

ELEVATION_LIMITS = (100, 400)  # [m] Limits for all plots

# UTM (need to adjust to show all stations)
XLIM = (336800, 337500)
YLIM = (7839600, 7840300)

# gdalwarp -t_srs EPSG:32759 -r cubicspline DEM_WGS84.tif DEM_WGS84_UTM.tif
DEM_FILE = WORKING_DIR / 'data' / 'DEM_WGS84_UTM.tif'
# DEM_FILE = WORKING_DIR / 'data' / 'DEM_Union_UAV_161116_sm101_UTM.tif'

# Read in full-res DEM, clip to extent to reduce size
dem = xr.open_rasterio(DEM_FILE).squeeze()
dem = dem.where(dem != dem.nodatavals)  # Set no data values to NaN
dem = dem.where(
    (dem.x >= XLIM[0]) & (dem.x <= XLIM[1]) & (dem.y >= YLIM[0]) & (dem.y <= YLIM[1])
)

# Plot DEM
fig_dem, ax_dem = plt.subplots()
dem.plot.imshow(
    ax=ax_dem,
    cmap='Greys_r',
    cbar_kwargs=dict(label='Elevation (m)'),
    add_labels=False,
    vmin=ELEVATION_LIMITS[0],
    vmax=ELEVATION_LIMITS[1],
)
ax_dem.set_aspect('equal')
ax_dem.ticklabel_format(style='plain', useOffset=False)
ax_dem.set_xlabel('UTM easting (m)')
ax_dem.set_ylabel('UTM northing (m)')
ax_dem.set_xlim(XLIM)
ax_dem.set_ylim(YLIM)
ax_dem.xaxis.set_ticks_position('both')
ax_dem.yaxis.set_ticks_position('both')
for label in ax_dem.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')
fig_dem.tight_layout()
fig_dem.show()

# Vent locations in UTM
x_A, y_A, *_ = utm.from_latlon(*VENT_LOCS['A'][::-1])
x_C, y_C, *_ = utm.from_latlon(*VENT_LOCS['C'][::-1])

# Station locations in UTM, read from labeled waveform file
STATION_COORDS = {}
for tr in read(str(WORKING_DIR / 'data' / 'labeled' / 'label_000.pkl'))[:5]:
    x, y, *_ = utm.from_latlon(tr.stats.latitude, tr.stats.longitude)
    STATION_COORDS[tr.stats.station] = x, y

# Actually interpolate!
profiles_A = []
profiles_C = []
N = 500  # Number of points in profile (overkill!)
for station_coord in STATION_COORDS.values():
    profile_A = dem.interp(
        x=xr.DataArray(np.linspace(x_A, station_coord[0], N)),
        y=xr.DataArray(np.linspace(y_A, station_coord[1], N)),
        method='linear',
    )
    profiles_A.append(profile_A)
    profile_C = dem.interp(
        x=xr.DataArray(np.linspace(x_C, station_coord[0], N)),
        y=xr.DataArray(np.linspace(y_C, station_coord[1], N)),
        method='linear',
    )
    profiles_C.append(profile_C)

vent_marker_kwargs = dict(color='white', edgecolor='black', zorder=5)
station_marker_kwargs = dict(marker='v', edgecolor='black', zorder=5)

# Plot profiles as groups of lines
fig, axes = plt.subplots(ncols=2, sharey=True)
for ax, profiles in zip(axes, [profiles_A, profiles_C]):
    for p, name in zip(profiles, STATION_COORDS.keys()):
        h = np.hstack(
            [0, np.cumsum(np.linalg.norm([np.diff(p.x), np.diff(p.y)], axis=0))]
        )
        l = ax.plot(h, p)
        ax.scatter(
            h[-1], p[-1], color=l[0].get_color(), label=name, **station_marker_kwargs
        )
    ax.scatter(0, p[0], label='Vent', clip_on=False, **vent_marker_kwargs)
    ax.set_aspect('equal')
    ax.set_ylim(*ELEVATION_LIMITS)
    ax.set_xlim(0, 450)
    ax.set_xlabel('Horizontal distance (m)')
    ax.xaxis.set_minor_locator(MultipleLocator(50))  # Minor ticks every 50 m
axes[0].set_title('Vent A')
axes[1].set_title('Vent C')
axes[0].set_ylabel('Elevation (m)')
axes[1].legend()
fig.tight_layout()
fig.show()

# Plot profiles on DEM
for pA, pC in zip(profiles_A, profiles_C):
    l = ax_dem.plot(pA.x.values, pA.y.values)
    ax_dem.plot(pC.x.values, pC.y.values, color=l[0].get_color())
for name, station_coord in STATION_COORDS.items():
    ax_dem.scatter(*station_coord, **station_marker_kwargs)
    ax_dem.text(*station_coord, s='  ' + name, va='center')
ax_dem.scatter(x_A, y_A, **vent_marker_kwargs)
ax_dem.scatter(x_C, y_C, **vent_marker_kwargs)
fig_dem.show()

# fig_dem.savefig('/Users/ldtoney/Downloads/profiles_dem.png', bbox_inches='tight', dpi=300)
# fig.savefig('/Users/ldtoney/Downloads/profiles_lines.png', bbox_inches='tight', dpi=300)

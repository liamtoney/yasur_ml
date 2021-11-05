import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import utm
import xarray as xr
from matplotlib.colors import LightSource
from matplotlib.ticker import MultipleLocator
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# UTM axis limits (need to adjust to show all stations)
XLIM = (336800, 337500)
YLIM = (7839600, 7840300)

# Define new color cycle based on entries 3–7 in "New Tableau 10", see
# https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-56782
COLOR_CYCLE = ['#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']

# gdalwarp -t_srs EPSG:32759 -r cubicspline DEM_WGS84.tif DEM_WGS84_UTM.tif
DEM_FILE = WORKING_DIR / 'data' / 'DEM_WGS84_UTM.tif'  # ~15 cm res
# DEM_FILE = WORKING_DIR / 'data' / 'DEM_Union_UAV_161116_sm101_UTM.tif'  # ~2 m res

# Read in full-res DEM, clip to extent to reduce size
dem = xr.open_rasterio(DEM_FILE).squeeze()
dem = dem.where(dem != dem.nodatavals)  # Set no data values to NaN
dem = dem.where(
    (dem.x >= XLIM[0]) & (dem.x <= XLIM[1]) & (dem.y >= YLIM[0]) & (dem.y <= YLIM[1])
)

# Create hillshade
ls = LightSource()
hs = dem.copy()
hs.data = ls.hillshade(
    dem.data, dx=np.abs(np.diff(dem.x).mean()), dy=np.abs(np.diff(dem.y).mean())
)

# Plot DEM
fig_dem, ax_dem = plt.subplots()
hs.plot.imshow(
    ax=ax_dem,
    cmap='Greys_r',
    add_colorbar=False,
    add_labels=False,
    alpha=0.4,  # Balance between rich contrast and swamping the station markers / lines
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

# Station locations in UTM
net = Client('IRIS').get_stations(
    network='3E',
    station='YIF1,YIF2,YIF3,YIF4,YIF5',
    starttime=UTCDateTime(2016, 1, 1),
    endtime=UTCDateTime(2016, 12, 31),
    level='station',
)[0]
STATION_COORDS = {}
for sta in net:
    x, y, *_ = utm.from_latlon(sta.latitude, sta.longitude)
    STATION_COORDS[sta.code] = x, y

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
    for p, name, color in zip(profiles, STATION_COORDS.keys(), COLOR_CYCLE):
        h = np.hstack(
            [0, np.cumsum(np.linalg.norm([np.diff(p.x), np.diff(p.y)], axis=0))]
        )
        ax.plot(h, p, color=color)
        ax.scatter(h[-1], p[-1], color=color, label=name, **station_marker_kwargs)
    ax.scatter(0, p[0], label='Subcrater', clip_on=False, **vent_marker_kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('Horizontal distance (m)')
    major_int = 100  # [m]
    ax.xaxis.set_major_locator(MultipleLocator(major_int))
    ax.yaxis.set_major_locator(MultipleLocator(major_int))
    minor_int = 50  # [m]
    ax.xaxis.set_minor_locator(MultipleLocator(minor_int))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_int))
    ax.set_xlim(0, 450)
    ax.set_ylim(100, 400)
    grid_params = dict(
        color=plt.rcParams['grid.color'],
        linewidth=plt.rcParams['grid.linewidth'],
        linestyle=':',
        zorder=-1,
        alpha=0.5,
        clip_on=False,
    )
    for x in np.arange(*ax.get_xlim(), minor_int):
        ax.axvline(x=x, **grid_params)
    for y in np.arange(*ax.get_ylim(), minor_int):
        ax.axhline(y=y, **grid_params)
    for side in 'right', 'top':
        ax.spines[side].set_visible(False)
axes[0].set_title('Subcrater S')
axes[1].set_title('Subcrater N')
axes[0].set_ylabel('Elevation (m)')
fig.tight_layout()
fig.subplots_adjust(wspace=0.2)
fig.show()

# Hard-coded numbers controlling where along profile the distance text is placed,
# ranging from 0 (at subcrater) to 1 (at station)
PROF_FRAC = dict(
    YIF1=dict(A=0.13, C=0.18),
    YIF2=dict(A=0.75, C=0.5),
    YIF3=dict(A=0.5, C=0.73),
    YIF4=dict(A=0.5, C=0.5),
    YIF5=dict(A=0.5, C=0.5),
)
GAP_HALF_WIDTH = 30  # [m] Half of the width of the gap in the line (where text goes)

# Plot horizontal profiles on DEM, adding text denoting distances
for pA, pC, prof_frac, color in zip(
    profiles_A, profiles_C, PROF_FRAC.values(), COLOR_CYCLE
):

    for profile, vent in zip([pA, pC], ['A', 'C']):

        # Convert profile x and y into masked arrays
        p_x = np.ma.array(profile.x.values)
        p_y = np.ma.array(profile.y.values)

        # Calculate full and component-wise lengths
        xlen = p_x[-1] - p_x[0]
        ylen = p_y[-1] - p_y[0]
        length = np.linalg.norm([xlen, ylen])

        center_ind = int(prof_frac[vent] * N)  # Convert fraction to index along profile
        ext_ind = int(GAP_HALF_WIDTH * N / length)  # Convert gap length to index

        # Mask profile arrays - masked entries don't get plotted!
        p_slice = slice(center_ind - ext_ind, center_ind + ext_ind)
        p_x[p_slice] = np.ma.masked
        p_y[p_slice] = np.ma.masked

        # Plot horizontal profile w/ gaps
        ax_dem.plot(p_x, p_y, color=color, linewidth=1)

        # Plot angled text showing distance along each path in meters
        ax_dem.text(
            p_x.data[center_ind],
            p_y.data[center_ind],
            f'{length:.0f} m',
            rotation=np.rad2deg(np.arctan(ylen / xlen)),
            va='center',
            ha='center',
            color=color,
            weight='bold',
            fontsize='6.5',
        )

for (name, station_coord), color in zip(STATION_COORDS.items(), COLOR_CYCLE):
    ax_dem.scatter(*station_coord, color=color, **station_marker_kwargs)
    ax_dem.text(*station_coord, s='  ' + name, va='center')
ax_dem.scatter(x_A, y_A, **vent_marker_kwargs)
ax_dem.scatter(x_C, y_C, **vent_marker_kwargs)
fig_dem.show()

# fig_dem.savefig('/Users/ldtoney/Downloads/profiles_dem.png', bbox_inches='tight', dpi=300)
# fig.savefig('/Users/ldtoney/Downloads/profiles_lines.png', bbox_inches='tight', dpi=300)

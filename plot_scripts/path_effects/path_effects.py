import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import utm
import xarray as xr
from matplotlib import transforms
from matplotlib.colors import LightSource
from matplotlib.ticker import MultipleLocator
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from svm import COLOR_CYCLE
from svm.plotting import plot_path_effect_matrix

FONT_SIZE = 14  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# Load station locations in UTM
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

# Determine DEM plot axis limits from station coords, add additional padding [m]
sta_x, sta_y = np.array(list(STATION_COORDS.values())).T
dem_xlim = np.array([sta_x.min() - 70, sta_x.max() + 30])
dem_ylim = np.array([sta_y.min() - 30, sta_y.max() + 20])

# gdalwarp -t_srs EPSG:32759 -r cubicspline DEM_WGS84.tif DEM_WGS84_UTM.tif
DEM_FILE = WORKING_DIR / 'data' / 'DEM_WGS84_UTM.tif'  # ~15 cm res
# DEM_FILE = WORKING_DIR / 'data' / 'DEM_Union_UAV_161116_sm101_UTM.tif'  # ~2 m res

# Common linewidth for profile lines
PROFILE_LW = 1.5

# Common major and minor tick intervals for plots [m]
MAJOR_INT = 100
MINOR_INT = 50

# Load scores to plot for panel (a)
SCORE_FILE = '2016-08-01.npy'
scores = np.load(WORKING_DIR / 'plot_scripts' / 'path_effects' / SCORE_FILE)

# Read in full-res DEM, clip to extent to reduce size
dem = xr.open_rasterio(DEM_FILE).squeeze()
dem = dem.where(dem != dem.nodatavals)  # Set no data values to NaN
dem = dem.where(
    (dem.x >= dem_xlim[0])
    & (dem.x <= dem_xlim[1])
    & (dem.y >= dem_ylim[0])
    & (dem.y <= dem_ylim[1])
)

# Create hillshade
ls = LightSource()
hs = dem.copy()
hs.data = ls.hillshade(
    dem.data, dx=np.abs(np.diff(dem.x).mean()), dy=np.abs(np.diff(dem.y).mean())
)

# Vent locations in UTM
x_A, y_A, *_ = utm.from_latlon(*VENT_LOCS['A'][::-1])
x_C, y_C, *_ = utm.from_latlon(*VENT_LOCS['C'][::-1])

# Calculate vertical profiles
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

#%% Plot

fig = plt.figure(figsize=(10.5, 13.5))
gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[3, 0.1, 3])

# --------------------------------------------------------------------------------------
# Panel (a)
# --------------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[1, 0])

plot_path_effect_matrix(
    scores,
    fig,
    ax1,
    day=UTCDateTime(SCORE_FILE.rstrip('.npy')),
    colorbar=cax,
    show_stats=False,
    diagonal_metrics=True,
)

# --------------------------------------------------------------------------------------
# Panel (b)
# --------------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[:1, 1])

# Plot DEM
hs.plot.imshow(
    ax=ax2,
    cmap='Greys_r',
    add_colorbar=False,
    add_labels=False,
    alpha=0.4,  # Balance between rich contrast and swamping the station markers / lines
)
ax2.set_aspect('equal')
ax2.set_xlim(dem_xlim)
ax2.set_ylim(dem_ylim)
ax2.axis('off')

# Add north arrow
x = 0.15
y = 0.9
arrow_length = 0.1
ax2.annotate(
    'N',
    xy=(x, y),
    xytext=(x, y - arrow_length),
    ha='center',
    arrowprops=dict(
        edgecolor='none',
        facecolor='black',
        arrowstyle='wedge,tail_width=0.6',
    ),
    xycoords='axes fraction',
    weight='bold',
)

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
        ax2.plot(p_x, p_y, color=color, linewidth=PROFILE_LW)

        # Plot angled text showing distance along each path in meters
        ax2.text(
            p_x.data[center_ind],
            p_y.data[center_ind],
            f'{length:.0f} m',
            rotation=np.rad2deg(np.arctan(ylen / xlen)),
            va='center',
            ha='center',
            color=color,
            weight='bold',
            fontsize='8',
        )

vent_marker_kwargs = dict(s=80, color='white', edgecolor='black', zorder=5)
station_marker_kwargs = dict(s=80, marker='v', edgecolor='black', zorder=5)

for (name, station_coord), color in zip(STATION_COORDS.items(), COLOR_CYCLE):
    ax2.scatter(*station_coord, color=color, **station_marker_kwargs)
ax2.scatter(x_A, y_A, **vent_marker_kwargs)
ax2.scatter(x_C, y_C, **vent_marker_kwargs)

# --------------------------------------------------------------------------------------
# Panels (c,d)
# --------------------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1], sharey=ax3)

CD_XLIM = (0, 450)  # [m]

for ax, profiles in zip([ax3, ax4], [profiles_A, profiles_C]):
    for p, name, color in zip(profiles, STATION_COORDS.keys(), COLOR_CYCLE):
        h = np.hstack(
            [0, np.cumsum(np.linalg.norm([np.diff(p.x), np.diff(p.y)], axis=0))]
        )
        ax.plot(h, p, color=color, linewidth=PROFILE_LW)
        ax.scatter(h[-1], p[-1], color=color, label=name, **station_marker_kwargs)
    ax.scatter(0, p[0], label='Subcrater', clip_on=False, **vent_marker_kwargs)
    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(MAJOR_INT))
    ax.yaxis.set_major_locator(MultipleLocator(MAJOR_INT))
    ax.xaxis.set_minor_locator(MultipleLocator(MINOR_INT))
    ax.yaxis.set_minor_locator(MultipleLocator(MINOR_INT))
    ax.set_xlim(0, np.diff(ax2.get_xlim())[0])  # Initially set xlim equal to DEM
    ax.set_ylim(100, 400)
    grid_params = dict(
        color=plt.rcParams['grid.color'],
        linewidth=plt.rcParams['grid.linewidth'],
        linestyle=':',
        zorder=-1,
        alpha=0.5,
        clip_on=False,
    )
    for x in np.arange(*CD_XLIM, MINOR_INT):
        ax.axvline(x=x, **grid_params)
    for y in np.arange(*ax.get_ylim(), MINOR_INT):
        ax.axhline(y=y, **grid_params)
    for side in 'right', 'top':
        ax.spines[side].set_visible(False)

pad = 10
ax3.set_xlabel('Distance from subcrater S (m)', labelpad=pad)
ax4.set_xlabel('Distance from subcrater N (m)', labelpad=pad)
ax3.set_ylabel('Elevation (m)', labelpad=15)
ax4.tick_params(which='both', labelleft=False)

# --------------------------------------------------------------------------------------
# Adjustments
# --------------------------------------------------------------------------------------

# Adjust cax
pos1 = ax1.get_position()
cbar = cax.get_position()
cax.set_position([pos1.x0, pos1.y0 - (2.5 * cbar.height), cbar.width, cbar.height])

# Adjust ax2
pos2 = ax2.get_position()
ax2.set_position([pos2.x0, pos1.ymax - pos2.height, pos2.width, pos2.height])

# Adjust ax3, ax4
for ax in ax3, ax4:
    pos = ax.get_position()
    ax.set_position(pos)  # Lock
    ax.set_xlim(CD_XLIM)
yoff = 0.12
pos3 = ax3.get_position()
ax3.set_position([pos1.xmax - pos3.width, pos3.y0 + yoff, pos3.width, pos3.height])
pos4 = ax4.get_position()
ax4.set_position([pos2.x0, pos4.y0 + yoff, pos4.width, pos4.height])

# Plot (a), (b), (c), (d) tags
text_kwargs = dict(x=-0.075, y=1, ha='right', weight='bold', fontsize=18)
t3_trans = transforms.blended_transform_factory(ax1.transAxes, ax3.transAxes)
t4_trans = transforms.blended_transform_factory(ax2.transAxes, ax4.transAxes)
ax1.text(s='A', va='bottom', transform=ax1.transAxes, **text_kwargs)
ax2.text(s='B', va='bottom', transform=ax2.transAxes, **text_kwargs)
ax3.text(s='C', va='top', transform=t3_trans, **text_kwargs)
ax4.text(s='D', va='top', transform=t4_trans, **text_kwargs)

fig.show()

_ = subprocess.run(['open', os.environ['YASUR_FIGURE_DIR']])

# fig.savefig(Path(os.environ['YASUR_FIGURE_DIR']) / 'path_effects.png', bbox_inches='tight', dpi=300)

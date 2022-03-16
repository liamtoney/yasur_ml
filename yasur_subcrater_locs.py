#!/usr/bin/env python

"""
Generates JSON file which contains estimated locations for subcrater S and subcrater N
based upon DEM minima for each subcrater (also calculates the midpoint between the two
subcrater). Locations are WGS84 (longitude, latitude).
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()
DEM_FILE = WORKING_DIR / 'data' / 'DEM_WGS84.tif'

# Read in DEM
dem = xr.open_rasterio(DEM_FILE).squeeze()

# Mask zero values
dem = dem.where(dem != 0)

# Plot a nice view of the crater area
fig, ax = plt.subplots()
dem.plot.imshow(ax=ax, cmap='Greys_r')
ax.set_xlim(169.4449, 169.4509)
ax.set_ylim(-19.5322, -19.5253)

# The below limits were obtained from conservatively windowing each subcrater
SUBCRATER_S = dict(
    xlim=(169.4463818548387, 169.44947409274192),
    ylim=(-19.530814772727275, -19.528406493506495),
)
SUBCRATER_N = dict(
    xlim=(169.4471152217742, 169.44930776209677),
    ylim=(-19.528621185064935, -19.52670762987013),
)

# Find minimum for each subcrater
locations = {}
for subcrater, label in zip([SUBCRATER_S, SUBCRATER_N], ['S', 'N']):

    # Crop to above limits
    mask_lon = (dem.x > subcrater['xlim'][0]) & (dem.x < subcrater['xlim'][1])
    mask_lat = (dem.y > subcrater['ylim'][0]) & (dem.y < subcrater['ylim'][1])
    cropped = dem.where(mask_lon & mask_lat, drop=True)

    # Find minimum
    minimum = cropped.where(cropped == cropped.min(), drop=True).squeeze()
    x_min = minimum.x.values.tolist()
    y_min = minimum.y.values.tolist()

    # Store
    locations[label] = [x_min, y_min]

# Calculate midpoint
x_mid = np.mean([locations['S'][0], locations['N'][0]])
y_mid = np.mean([locations['S'][1], locations['N'][1]])
locations['midpoint'] = [x_mid, y_mid]

# Plot points
for label, loc in locations.items():
    try:
        color = os.environ[f'SUBCRATER_{label}']
    except KeyError:  # For midpoint
        color = 'white'
    ax.scatter(loc[0], loc[1], s=70, label=label, color=color, edgecolors='black')
ax.legend()

# Show figure
fig.show()

# Write JSON file
with open(WORKING_DIR / 'yasur_subcrater_locs.json', 'w') as f:
    json.dump(locations, f, indent=2)
    f.write('\n')

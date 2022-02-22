#!/usr/bin/env python

"""
Read in CSV file corresponding to a catalog, fit a two-component Gaussian mixed model to
the catalog locations, and label events based upon their inclusion inside the confidence
ellipses of the two Gaussian distributions.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import utm
from matplotlib.patches import Ellipse
from obspy import Stream, UTCDateTime, read
from rtm import define_grid, produce_dem
from sklearn import mixture

# Toggle plotting
PLOT = False

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Define which catalog to label
catalog_csv = WORKING_DIR / 'label' / 'catalogs' / 'height_4_spacing_30_agc_60.csv'

# Number of standard deviations from mean of distribution to allow for subcrater
# association
N_STD = 2

# Read in entire catalog to pandas DataFrame
df = pd.read_csv(catalog_csv)
df.t = [UTCDateTime(t) for t in df.t]

# Load subcrater locs
with open(WORKING_DIR / 'yasur_subcrater_locs.json') as f:
    SUBCRATER_LOCS = json.load(f)

# Convert subcrater midpoint to UTM
x_0, y_0, *_ = utm.from_latlon(
    SUBCRATER_LOCS['midpoint'][1], SUBCRATER_LOCS['midpoint'][0]
)

# Need to define this for proper histogram binning (TODO: NEED TO UPDATE THIS IF build_catalog.py CHANGES!)
grid = define_grid(
    lon_0=SUBCRATER_LOCS['midpoint'][0],
    lat_0=SUBCRATER_LOCS['midpoint'][1],
    x_radius=350,
    y_radius=350,
    spacing=10,
    projected=True,
)
dem = produce_dem(
    grid,
    external_file=str(WORKING_DIR / 'data' / 'DEM_WGS84.tif'),
    plot_output=False,
)

#%% Define confidence ellipse function


def confidence_ellipse_from_mean_cov(
    mean, cov, ax, n_std=3.0, facecolor='none', **kwargs
):
    """
    Create a plot of the covariance confidence ellipse given mean and cov matrix.
    Modified from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    mean : 2D mean (mean_x, mean_y)
    cov : 2 x 2 cov matrix
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`
    Returns
    -------
    matplotlib.patches.Ellipse
    """

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(*mean)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


#%% Fit Gaussian mixture model (GMM)

# Format for scikit-learn
X_train = np.column_stack([df.x, df.y])

# Get UTM locations of the subcrater DEM minima to initialize the GMM
subcrater_utm = {}
for subcrater in 'S', 'N':
    subcrater_utm[subcrater] = utm.from_latlon(
        SUBCRATER_LOCS[subcrater][1], SUBCRATER_LOCS[subcrater][0]
    )[:2]

# Fit a GMM with two components (for the two subcraters)
clf = mixture.GaussianMixture(
    n_components=2,
    covariance_type='full',
    means_init=list(subcrater_utm.values()),
    random_state=47,  # We want reproducible results so the ellipses stay the same!
)
clf.fit(X_train)

# MUST plot since confidence ellipse function is designed to work in plotting context
fig, ax = plt.subplots()
ax.scatter(df.x, df.y, s=1, c='black')
for mean, cov, subcrater in zip(clf.means_, clf.covariances_, subcrater_utm.keys()):
    confidence_ellipse_from_mean_cov(
        mean, cov, ax, n_std=N_STD, edgecolor=os.environ[f'SUBCRATER_{subcrater}']
    )
in_ell = (
    {}
)  # KEY variable storing whether or not locs are within either subcrater ellipse
for ell, subcrater in zip(ax.patches, subcrater_utm.keys()):
    in_ell[subcrater] = ell.contains_points(
        ax.transData.transform(np.column_stack([df.x, df.y]))
    )
    ax.scatter(
        df.x[in_ell[subcrater]],
        df.y[in_ell[subcrater]],
        s=1,
        c=os.environ[f'SUBCRATER_{subcrater}'],
    )
    # Write out text file of verts (m from subcrater midpoint) for station map
    verts = ax.transData.inverted().transform(ell.get_verts())
    verts[:, 0] -= x_0
    verts[:, 1] -= y_0
    np.savetxt(
        WORKING_DIR / 'plot_scripts' / 'station_map' / f'{subcrater}_ellipse.xy',
        verts,
        fmt='%.4f',
    )

    print(f'Subcrater {subcrater}: {in_ell[subcrater].sum()}')
print('Total: {}'.format(in_ell['S'].sum() + in_ell['N'].sum()))
ax.set_aspect('equal')
fig.show()

#%% (OPTIONAL) Plot histogram of catalog with ellipses overlain

if PLOT:

    # Change registration
    xe = np.hstack([grid.x.values, grid.x.values[-1] + grid.spacing]) - grid.spacing / 2
    ye = np.hstack([grid.y.values, grid.y.values[-1] + grid.spacing]) - grid.spacing / 2

    # Compute histogram and assign to DataArray
    h, *_ = np.histogram2d(df.x, df.y, bins=[xe, ye])
    h[h == 0] = np.nan
    hist = grid.copy()
    hist.data = h.T  # Because of NumPy array axis handling

    fig, ax = plt.subplots()
    dem.plot.contour(ax=ax, levels=20, colors='black', linewidths=0.5)
    hist.plot.pcolormesh(ax=ax, cmap='hot_r', cbar_kwargs=dict(label='# of events'))
    for mean, cov in zip(clf.means_, clf.covariances_):
        confidence_ellipse_from_mean_cov(mean, cov, ax, n_std=N_STD, edgecolor='black')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'{df.shape[0]} events in catalog, N_STD = {N_STD}')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')

    # Tick formatting
    RADIUS = 300
    INC = 100
    for axis, ref_coord in zip([ax.xaxis, ax.yaxis], [x_0, y_0]):
        fmt = axis.get_major_formatter()
        fmt.set_useOffset(ref_coord)
        axis.set_major_formatter(fmt)
        plt.setp(axis.get_offset_text(), visible=False)
        axis.set_ticks(np.arange(-RADIUS, RADIUS + INC, INC) + ref_coord)

    fig.show()

#%% Label based upon whether location is inside / outside confidence ellipse

# Array of subcrater labels for each entry in catalog
subcrater_locs = np.empty(df.shape[0], dtype=str)
for subcrater, in_ell_ind in in_ell.items():
    subcrater_locs[in_ell_ind] = subcrater
subcrater_locs[
    in_ell['S'] & in_ell['N']
] = ''  # Doubly-located events should be discarded

df['subcrater'] = subcrater_locs

# Remove rows with no location
df_locs = df[df.subcrater != '']
df_locs.reset_index(inplace=True, drop=True)  # Important since iterrows() uses index!

#%% Load in full dataset

print('Reading in full dataset...')
st_full = read(str(WORKING_DIR / 'data' / '3E_YIF1-5_50hz.pkl'))
print('Done')

WAVEFORM_DUR = 5  # [s] Duration of labeled waveform snippets

fs = st_full[0].stats.sampling_rate  # [Hz]

length_samples = int(WAVEFORM_DUR * fs)  # [samples]

print('Labeling waveforms...')
n = 0
st_label = Stream()
for i, row in df_locs.iterrows():

    st = st_full.copy().trim(row.t, row.t + WAVEFORM_DUR)  # TODO: This line is slow!
    for tr in st:
        tr.stats.subcrater = row.subcrater
        tr.stats.event_info = dict(utm_x=row.x, utm_y=row.y, origin_time=row.t)
        tr.data = tr.data[:length_samples]

    st_label += st

    if (i + 1) % 10 == 0:
        st_label.write(
            str(WORKING_DIR / 'data' / 'labeled' / f'label_{n:03}.pkl'),
            format='PICKLE',
        )
        st_label = Stream()
        print(f'{((i + 1) / df_locs.shape[0]) * 100:.2f}%')
        n += 1

# Handle last one
if st_label.count() > 0:
    st_label.write(
        str(WORKING_DIR / 'data' / 'labeled' / f'label_{n:03}.pkl'), format='PICKLE'
    )

print('Done')

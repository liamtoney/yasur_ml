#!/usr/bin/env python

"""
Read in CSV file corresponding to a catalog, fit a two-component Gaussian mixed model to
the catalog locations, and label events based upon their inclusion inside the confidence
ellipses of the two Gaussian distributions.
"""

import json
import os
from pathlib import Path

import colorcet as cc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import utm
from matplotlib.patches import Ellipse
from matplotlib.ticker import PercentFormatter
from obspy import Stream, UTCDateTime, read
from rtm import define_grid, produce_dem
from sklearn import mixture

# Toggle plotting
PLOT = False

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Define which catalog to label
catalog_csv = WORKING_DIR / 'label' / 'catalogs' / 'height_4_spacing_30_agc_60.csv'

# Number of standard deviations from mean of distribution to allow for vent association
N_STD = 2

# Read in entire catalog to pandas DataFrame
df = pd.read_csv(catalog_csv)
df.t = [UTCDateTime(t) for t in df.t]

# Load vent locs
with open(WORKING_DIR / 'yasur_vent_locs.json') as f:
    VENT_LOCS = json.load(f)

# Convert vent midpoint to UTM
x_0, y_0, *_ = utm.from_latlon(VENT_LOCS['midpoint'][1], VENT_LOCS['midpoint'][0])

# Need to define this for proper histogram binning (TODO: NEED TO UPDATE THIS IF build_catalog.py CHANGES!)
grid = define_grid(
    lon_0=VENT_LOCS['midpoint'][0],
    lat_0=VENT_LOCS['midpoint'][1],
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

# Get UTM locations of the vent DEM minima to initialize the GMM
vent_utm = {}
for vent in 'A', 'C':
    vent_utm[vent] = utm.from_latlon(VENT_LOCS[vent][1], VENT_LOCS[vent][0])[:2]

# Fit a GMM with two components (for the two vents)
clf = mixture.GaussianMixture(
    n_components=2, covariance_type='full', means_init=list(vent_utm.values())
)
clf.fit(X_train)

# MUST plot since confidence ellipse function is designed to work in plotting context
fig, ax = plt.subplots()
ax.scatter(df.x, df.y, s=1, c='black')
for mean, cov, vent in zip(clf.means_, clf.covariances_, vent_utm.keys()):
    confidence_ellipse_from_mean_cov(
        mean, cov, ax, n_std=N_STD, edgecolor=os.environ[f'VENT_{vent}']
    )
in_ell = {}  # KEY variable storing whether or not locs are within either vent ellipse
for ell, vent in zip(ax.patches, vent_utm.keys()):
    in_ell[vent] = ell.contains_points(
        ax.transData.transform(np.column_stack([df.x, df.y]))
    )
    ax.scatter(
        df.x[in_ell[vent]], df.y[in_ell[vent]], s=1, c=os.environ[f'VENT_{vent}']
    )
    # Write out text file of verts (m from vent midpoint) for station map
    verts = ax.transData.inverted().transform(ell.get_verts())
    verts[:, 0] -= x_0
    verts[:, 1] -= y_0
    np.savetxt(WORKING_DIR / 'plot_scripts' / f'{vent}_ellipse.xy', verts, fmt='%.4f')

    print(f'Vent {vent}: {in_ell[vent].sum()}')
print('Total: {}'.format(in_ell['A'].sum() + in_ell['C'].sum()))
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
    hist.plot.pcolormesh(ax=ax, cmap=cc.m_fire_r, cbar_kwargs=dict(label='# of events'))
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

# Array of vent labels for each entry in catalog
vent_locs = np.empty(df.shape[0], dtype=str)
for vent, in_ell_ind in in_ell.items():
    vent_locs[in_ell_ind] = vent
vent_locs[in_ell['A'] & in_ell['C']] = ''  # Doubly-located events should be discarded

df['vent'] = vent_locs

# Remove rows with no location
df_locs = df[df.vent != '']
df_locs.reset_index(inplace=True, drop=True)  # Important since iterrows() uses index!

#%% (OPTIONAL) Make area plot of labeled catalog

if PLOT:

    # Toggle plotting fraction of vent A vs. vent C in each window (otherwise plot totals)
    FRACTION = False

    # [s] Rolling window duration
    WINDOW = 60 * 60

    # Start and end on whole hours
    t_start = UTCDateTime('2016-07-27T05')
    t_end = UTCDateTime('2016-08-01T22')

    # Form array of UTCDateTimes
    t_vec = [t_start + t for t in np.arange(0, (t_end - t_start), WINDOW)]

    # In moving windows, get counts of vent A and vent C
    fraction_A = []
    fraction_C = []
    for t in t_vec:
        df_hr = df_locs[(df_locs.t >= t) & (df_locs.t < t + WINDOW)]
        vcounts = df_hr.vent.value_counts()
        if FRACTION:
            vcounts /= vcounts.sum()
        if hasattr(vcounts, 'A'):
            fraction_A.append(vcounts.A)
        else:
            fraction_A.append(0)
        if hasattr(vcounts, 'C'):
            fraction_C.append(vcounts.C)
        else:
            fraction_C.append(0)

    # Load in a single station's data and process (takes a while, can comment out for repeat
    # runs)
    tr = read(str(WORKING_DIR / 'data' / '3E_YIF1-5_50hz.pkl')).select(station='YIF3')[
        0
    ]
    tr.remove_response()
    tr.filter('bandpass', freqmin=0.2, freqmax=4, zerophase=True)

    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(13, 5))

    # Subplot 1: Waveform
    axes[0].plot(tr.times('matplotlib'), tr.data, linewidth=0.5, color='black')
    axes[0].set_ylabel('Pressure (Pa)')

    # Subplot 2: Stacked area plot
    t_vec_mpl = [t.matplotlib_date for t in t_vec]
    axes[1].stackplot(
        t_vec_mpl,
        fraction_A,
        fraction_C,
        colors=(os.environ['VENT_A'], os.environ['VENT_C']),
        labels=('Subcrater S', 'Subcrater N'),
    )
    if FRACTION:
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    else:
        axes[1].set_ylabel('Number of labeled events')
    axes[1].autoscale(enable=True, axis='y', tight=True)

    # Overall x-axis formatting
    axes[-1].set_xlim(t_start.matplotlib_date, (t_end - WINDOW).matplotlib_date)
    loc = axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    # Add legend
    axes[-1].legend(loc='lower right')

    fig.show()

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

    st = st_full.copy().trim(row.t, row.t + WAVEFORM_DUR)  # TODO: This line is slow(?)
    for tr in st:
        tr.stats.vent = row.vent
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

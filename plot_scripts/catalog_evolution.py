#!/usr/bin/env python

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from obspy import UTCDateTime, read

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Toggle plotting fraction of vent A vs. vent C in each window (otherwise plot totals)
FRACTION = False

# [s] Rolling window duration
WINDOW = 60 * 60

# Read in features CSV file, process into a catalog by subsetting to just one station
# and removing the unneeded features columns
catalog = pd.read_csv(WORKING_DIR / 'features' / 'csv' / 'features_highpass.csv')
catalog = catalog[catalog.station == 'YIF1'][['time', 'label']]
catalog.time = [UTCDateTime(t) for t in catalog.time]

# Start and end on whole hours
t_start = UTCDateTime('2016-07-27T05')
t_end = UTCDateTime('2016-08-01T22')

# Form array of UTCDateTimes
t_vec = [t_start + t for t in np.arange(0, (t_end - t_start), WINDOW)]

# In moving windows, get counts of vent A and vent C
fraction_A = []
fraction_C = []
for t in t_vec:
    catalog_hr = catalog[(catalog.time >= t) & (catalog.time < t + WINDOW)]
    vcounts = catalog_hr.label.value_counts()
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

# Load in a single station's data and process (takes a while, disable for repeat runs)
if True:
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

# fig.savefig(Path(os.environ['YASUR_FIGURE_DIR']) / 'catalog_evolution.png', bbox_inches='tight', dpi=300)

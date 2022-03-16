#!/usr/bin/env python

import os
import subprocess
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read

# Define project directory
WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()

FONT_SIZE = 14  # [pt]
plt.rcParams.update({'font.size': FONT_SIZE})

# [s] Rolling window duration
WINDOW = 60 * 60

# Read in features file, process into a catalog by subsetting to just one station and
# removing the unneeded features columns
catalog = pd.read_feather(
    WORKING_DIR / 'features' / 'feather' / 'manual_filter_roll.feather'
)
catalog = catalog[catalog.station == 'YIF1'][['time', 'label']]
catalog.time = [UTCDateTime(t) for t in catalog.time]

# Start and end on whole hours
t_start = UTCDateTime('2016-07-27T05')
t_end = UTCDateTime('2016-08-01T22')

# Form array of UTCDateTimes
t_vec = [t_start + t for t in np.arange(0, (t_end - t_start), WINDOW)]

# In moving windows, get counts of subcrater S and subcrater N
fraction_S = []
fraction_N = []
for t in t_vec:
    catalog_hr = catalog[(catalog.time >= t) & (catalog.time < t + WINDOW)]
    vcounts = catalog_hr.label.value_counts()
    if hasattr(vcounts, 'S'):
        fraction_S.append(vcounts.S)
    else:
        fraction_S.append(0)
    if hasattr(vcounts, 'N'):
        fraction_N.append(vcounts.N)
    else:
        fraction_N.append(0)

# Load in a single station's data and process (takes a while)
tr = read(str(WORKING_DIR / 'data' / '3E_YIF1-5_50hz.pkl')).select(station='YIF3')[0]
tr.remove_response()
tr.filter('bandpass', freqmin=0.2, freqmax=4, zerophase=True)

#%% Create plot

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(13, 5))

# Subplot 1: Waveform
tr_trim = tr.copy().trim(t_vec[0] + (WINDOW / 2), t_vec[-1] + (WINDOW / 2))
axes[0].plot(
    tr_trim.times('matplotlib'),
    tr_trim.data,
    linewidth=0.5,
    color='black',
    clip_on=False,
    solid_capstyle='round',
)
axes[0].set_ylabel('Pressure (Pa)')
axes[0].set_ylim(-500, 500)
axes[0].tick_params(axis='x', which='both', bottom=False)

# Subplot 2: Stacked area plot
t_vec_mpl = [(t + (WINDOW / 2)).matplotlib_date for t in t_vec]  # Center in window!
axes[1].stackplot(
    t_vec_mpl,
    fraction_S,
    fraction_N,
    colors=(os.environ['SUBCRATER_S'], os.environ['SUBCRATER_N']),
    labels=('S subcrater', 'N subcrater'),
    clip_on=False,
)
axes[1].set_ylabel('# of labeled events')
axes[1].set_ylim(0, 70)
axes[1].yaxis.set_minor_locator(plt.MultipleLocator(10))

# Remove spines
for side in 'bottom', 'top', 'right':
    axes[0].spines[side].set_visible(False)
for side in 'top', 'right':
    axes[1].spines[side].set_visible(False)

# Overall x-axis formatting
axes[-1].set_xlim(t_vec_mpl[0], t_vec_mpl[-1])  # Bounds of the area plot
loc = axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%-d %B'))
axes[-1].xaxis.set_minor_locator(mdates.HourLocator(range(0, 24, 12)))

# Add legend
axes[-1].legend(loc='lower right', framealpha=0.6, fancybox=False, edgecolor='none')

# Make connected gridlines
for ax in axes:
    ax.patch.set_alpha(0)
grid_ax = fig.add_subplot(1, 1, 1, zorder=-1, sharex=axes[-1])
for spine in grid_ax.spines.values():
    spine.set_visible(False)
grid_ax.tick_params(
    which='both',
    left=False,
    labelleft=False,
    bottom=False,
    labelbottom=False,
)
grid_ax.grid(which='both', axis='x', linestyle=':', alpha=0.5)

# Plot (a) and (b) tags
for ax, label in zip(axes, ['(a)', '(b)']):
    ax.text(
        -0.07,
        1,
        label,
        ha='right',
        va='top',
        transform=ax.transAxes,
        weight='bold',
        fontsize=18,
    )

fig.tight_layout()
fig.show()

_ = subprocess.run(['open', os.environ['YASUR_FIGURE_DIR']])

# fig.savefig(Path(os.environ['YASUR_FIGURE_DIR']) / 'catalog_evolution.png', bbox_inches='tight', dpi=300)

#!/usr/bin/env python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, read

plt.rcParams.update({'font.size': 14})

# Define project directory
WORKING_DIR = Path.home() / 'work' / 'yasur_ml'

# Define paths for all labeled waveform files and randomly pick 3
wf_files = np.array(
    [str(p) for p in sorted((WORKING_DIR / 'data' / 'labeled').glob('*'))]
)
inds = np.random.choice(range(len(wf_files)), size=3, replace=False)

# Read in data
st = Stream()
for file in wf_files[inds]:
    st += read(file)

# Create a Stream containing 10 waveforms: Vent A and C for each of five stations
st_plot = Stream()
for station in 'YIF1', 'YIF2', 'YIF3', 'YIF4', 'YIF5':
    st_sta = st.select(station=station)
    vents = np.array([tr.stats.vent for tr in st_sta])
    rand_A_ind = np.random.choice(np.where(vents == 'A')[0])  # Random index for A
    rand_C_ind = np.random.choice(np.where(vents == 'C')[0])  # Random index for C
    st_plot += st_sta[rand_A_ind]
    st_plot += st_sta[rand_C_ind]

# Process
st_plot.remove_response()
st_plot.normalize()

# Plot
fig, ax = plt.subplots(figsize=(7, 9))
for i, tr in enumerate(st_plot):
    ax.plot(
        tr.times(),
        tr.data - 2.5 * i,  # Shifting down
        color=os.environ[f'VENT_{tr.stats.vent}'],
        solid_capstyle='round',
        clip_on=False,
    )
ax.autoscale(enable=True, axis='both', tight=True)
ax.set_xlim(0, 5)
ax.set_ylim(bottom=-2.5 * (i + 1))  # Add more room on bottom
ax.set_yticks([])
ax.set_xlabel('Time (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
lw = 1.2
ax.spines['bottom'].set_linewidth(lw)
ax.xaxis.set_tick_params(direction='in', pad=8, width=lw, length=5)
fig.tight_layout()
fig.show()

# fig.savefig('/Users/ldtoney/Downloads/wfs.png', bbox_inches='tight', dpi=300)

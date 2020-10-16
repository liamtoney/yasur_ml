"""
Download 3E data and save to pickle file(s).
"""

from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Stations of 3E to download data for
STATION = 'YIF1,YIF2,YIF3,YIF4,YIF5'

client = Client('IRIS')

net = client.get_stations(
    network='3E',
    station=STATION,
    starttime=UTCDateTime(2016, 1, 1),
    endtime=UTCDateTime(2016, 12, 31),
)[0]

# Find bounds of the time period where all stations were present
latest_start = np.max([sta.start_date for sta in net])
earliest_stop = np.min([sta.end_date for sta in net])

print(
    f'Downloading {(earliest_stop - latest_start) / (60 * 60):.1f} h of data for {len(net.stations)} channels'
)

# Download
st = client.get_waveforms(
    network='3E',
    station=STATION,
    location='01',
    channel='CDF',
    starttime=latest_start,
    endtime=earliest_stop,
    attach_response=True,
)

# Save (might need to downsample first? also might need to break into smaller files)
st.write(Path.home() / 'work' / 'yasur_ml' / 'data' / '3E_YIF1-5.pkl', format='PICKLE')

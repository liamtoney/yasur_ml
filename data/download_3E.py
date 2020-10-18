"""
Download 3E data and save to pickle file(s).
"""

from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

# Stations of 3E to download data for
STATION = 'YIF1,YIF2,YIF3,YIF4,YIF5'

# Duration of data chunks (for piecewise downloading)
CHUNK_DURATION = 0.5  # [h]

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

# Download in chunks (this will take a while!)
st = Stream()
starttime = latest_start
chunk_duration_sec = CHUNK_DURATION * 60 * 60  # [s]
num_hours_downloaded = 0  # [h]
while True:
    endtime = starttime + chunk_duration_sec
    leave = False
    if endtime > earliest_stop:
        endtime = earliest_stop
        leave = True
    st_chunk = client.get_waveforms(
        network='3E',
        station=STATION,
        location='01',
        channel='CDF',
        starttime=starttime,
        endtime=endtime,
        attach_response=True,
    )
    st += st_chunk
    if leave:
        break
    num_hours_downloaded += CHUNK_DURATION
    print(f'{num_hours_downloaded:g} hrs downloaded')
    starttime += chunk_duration_sec

# Save (might need to downsample first? also might need to break into smaller files)
# st.write(Path.home() / 'work' / 'yasur_ml' / 'data' / '3E_YIF1-5.pkl', format='PICKLE')

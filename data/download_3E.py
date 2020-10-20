"""
Download 3E data, downsample, add coordinates, and save to a pickle file.
"""

from pathlib import Path

import numpy as np
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn import Client

# Stations of 3E to download data for
STATION = 'YIF1,YIF2,YIF3,YIF4,YIF5'

# Duration of data chunks (for piecewise downloading)
CHUNK_DURATION = 0.5  # [h]

# New sampling rate to downsample to
SAMPLING_RATE = 50  # [Hz]

client = Client('IRIS')

net = client.get_stations(
    network='3E',
    station=STATION,
    starttime=UTCDateTime(2016, 1, 1),
    endtime=UTCDateTime(2016, 12, 31),
    level='channel',
)[0]

# Find bounds of the time period where all stations were present
latest_start = np.max([sta.start_date for sta in net])
earliest_stop = np.min([sta.end_date for sta in net])

print(
    f'Downloading {(earliest_stop - latest_start) / (60 * 60):.1f} h of data for {len(net.stations)} channels'
)

# Download in chunks (takes a while!)
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

# Downsample, and then merge (opposite order hangs)
st.interpolate(sampling_rate=SAMPLING_RATE, method='lanczos', a=20)
st.merge(method=1, fill_value='interpolate')  # Avoids creating a masked array

# Add coordinates
for tr in st:
    coords = net.get_coordinates(tr.id, datetime=latest_start)
    print(coords)
    tr.stats.latitude = coords['latitude']
    tr.stats.longitude = coords['longitude']
    tr.stats.elevation = coords['elevation']

# Save
filename = f'3E_YIF1-5_{SAMPLING_RATE}hz.pkl'
st.write(str(Path.home() / 'work' / 'yasur_ml' / 'data' / filename), format='PICKLE')

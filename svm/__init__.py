from obspy import UTCDateTime

# Deployment-related constants
ALL_STATIONS = [f'YIF{n}' for n in range(1, 6)]
ALL_DAYS = [
    UTCDateTime(2016, 7, 27),
    UTCDateTime(2016, 7, 28),
    UTCDateTime(2016, 7, 29),
    UTCDateTime(2016, 7, 30),
    UTCDateTime(2016, 7, 31),
    UTCDateTime(2016, 8, 1),
]

from . import plotting, tools

# Clean up
del UTCDateTime

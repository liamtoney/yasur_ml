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

# Define new color cycle based on entries 3–7 in "New Tableau 10", see
# https://www.tableau.com/about/blog/2016/7/colors-upgrade-tableau-10-56782
COLOR_CYCLE = ['#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1']

from . import plotting, tools

# Clean up
del UTCDateTime

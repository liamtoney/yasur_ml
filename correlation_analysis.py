import os
import pickle
from pathlib import Path

import numpy as np
from obspy import read

from svm import ALL_STATIONS

# Define project directory
WORKING_DIR = Path(os.environ['YASUR_WORKING_DIR']).expanduser().resolve()

# Bandpass filter corners [Hz] (must match those for template waveforms)
FREQMIN = 0.2
FREQMAX = 4

#%% Fetch all labeled waveforms; define correlation function

# This .pkl file is generated in catalog_waveforms.py
pickle_filename = (
    WORKING_DIR / 'plot_scripts' / 'catalog_waveforms' / f'traces_dict_filtered.pkl'
)
with pickle_filename.open('rb') as f:
    traces = pickle.load(f)


def norm_xcorr_zero_lag(a, b):
    """Compute normalized cross-correlation R_ab(tau) at zero time lag (tau = 0)."

    Args:
        a (numpy.ndarray): Time series "a"
        b (numpy.ndarray): Time series "b" (length must match length of "a")

    Returns:
        float: Correlation value R_ab(0), contained in the interval [-1, 1]
    """

    # Compute autocorrelations (needed for normalization)
    r_aa = np.correlate(a, a, mode='valid')
    r_bb = np.correlate(b, b, mode='valid')

    # Normalization factor
    norm_factor = 1 / np.sqrt(r_aa * r_bb)

    # Compute normalized cross-correlation & return as a float. 'valid' is key here; see
    # https://numpy.org/doc/stable/reference/generated/numpy.convolve.html#numpy.convolve
    r_ab_zero_lag = norm_factor * np.correlate(a, b, mode='valid')[0]
    return r_ab_zero_lag


#%% Loop over all labeled Streams, computing correlation with template waveforms

# Directory containing labeled waveforms
labeled_wf_dir = WORKING_DIR / 'data' / 'labeled'

# Initialize dictionary of accuracies
accuracy_dict = {'YIF1': [], 'YIF2': [], 'YIF3': [], 'YIF4': [], 'YIF5': []}

# Initialize dict of correlation coefficients (for later analysis)
all_coeffs = dict(S=[], N=[])

# Iterate over all labeled waveform files
for file in sorted(labeled_wf_dir.glob('label_???.pkl')):

    # Read in
    print(f'Reading {file}')
    st = read(str(file))

    # Process (filtering NEEDED since template waveforms are filtered)
    st.remove_response()
    st.detrend()
    st.taper(0.01)
    st.filter('bandpass', freqmin=FREQMIN, freqmax=FREQMAX, zerophase=True)
    st.normalize()

    # Go station-by-station
    for station in ALL_STATIONS:

        # Get waveforms for this station
        st_sta = st.select(station=station)

        # Initialize arrays of true and predicted subcrater labels
        y_true = []
        y_pred = []

        # Iterate over each Trace in the Stream
        for tr in st_sta:

            # Store coefficients for this Trace
            coeffs = {}

            # Correlate this Trace with each template waveform
            for test_subcrater in 'S', 'N':

                # Take median to get template (removing first row of NaNs here)
                vs_traces = traces[test_subcrater][station][1:, :]
                template = np.percentile(vs_traces, 50, axis=0)

                # KEY: Perform correlation to get coefficient
                coeff = norm_xcorr_zero_lag(tr.data, template)

                # Store coeff
                coeffs[test_subcrater] = coeff
                all_coeffs[test_subcrater].append(coeff)  # Storing to overall dict here

            # The predicted subcrater is just the subcrater with the higher coeff
            y_pred.append(max(coeffs, key=coeffs.get))
            y_true.append(tr.stats.subcrater)

        # Compute accuracy score and store
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        accuracy = (y_pred == y_true).sum() / y_true.size
        accuracy_dict[station].append(accuracy)

#%% Print accuracies for each station (compare to diagonal of Fig. 4a)

for station, accuracies in accuracy_dict.items():
    print(station)
    print(f'{np.mean(accuracies):.1%}\n')

#%% Analyze correlation coefficients

s_coeffs = np.array(all_coeffs['S'])
n_coeffs = np.array(all_coeffs['N'])

max_coeffs = np.max([s_coeffs, n_coeffs], axis=0)  # The higher value for each pair

# Print stats for the "winning" coefficients
print(f'Median: {np.median(max_coeffs)}:.2f')
print(f'Minimum: {max_coeffs.min():.2f}')
print(f'Maximum: {max_coeffs.max():.2f}')

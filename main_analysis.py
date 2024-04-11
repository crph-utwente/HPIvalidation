
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def analyse_data(dataframe, identifier, surgery_start_time, surgery_end_time):
    print(f'Analysing {identifier}')

    surgery_duration = surgery_end_time - surgery_start_time

    # make result dictionary
    result = {}
    result_vector = {}
    result['Patient ID'] = identifier
    result['Surgery duration (minutes)'] = surgery_duration.total_seconds() / 60

    data = pd.DataFrame(data=dataframe, columns=['datetime', 'MAP (mmHg)', 'HPI(-)'])
    data.columns = ['Time', 'MAP', 'HPI']

    # Crop dataframe to duration surgery
    data = data[(data['Time'] > surgery_start_time) & (data['Time'] < surgery_end_time)].reset_index(drop=True) 

    # Crop dataframe, when missing/invalid data from either HPI or MAP (at beginning or end)
    data = data[data['HPI'] != -1].reset_index(drop=True)
    data = data.loc[data['MAP'].first_valid_index():data['MAP'].last_valid_index()]
    data = data.loc[data['HPI'].first_valid_index():data['HPI'].last_valid_index()]

    # Reset index to make window detection work
    data = data.reset_index(drop=True)

    # Create clean data columns
    data['MAP_clean'] = data['MAP'].astype(np.float64)
    data['HPI_clean'] = data['HPI'].astype(np.float64)

    # Remove HPI & MAP values with a sudden MAP drop of 30mmHg compared to its surrounding values
    mask = (data['MAP_clean'].diff() <= 30 * -1) & (data['MAP_clean'].diff().shift(-1) >= 30)
    data.loc[mask, 'MAP_clean'] = np.nan
    data.loc[mask, 'HPI_clean'] = np.nan

    # Lineairly interpolate NaN values
    data['MAP_clean'] = data['MAP_clean'].interpolate(method='linear')
    data['HPI_clean'] = data['HPI_clean'].interpolate(method='linear')

    # Round interpolated values
    data['MAP_clean'] = data['MAP_clean'].round()
    data['HPI_clean'] = data['HPI_clean'].round()

    # Normalize signals HPI & MAP
    norm_HPI = (data['HPI_clean'] - np.mean(data['HPI_clean'])) / np.std(data['HPI_clean'])
    norm_MAP = (data['MAP_clean'] - np.mean(data['MAP_clean'])) / np.std(data['MAP_clean'])

    # Calculate the cross-correlation
    cross_corr = np.correlate(norm_HPI, norm_MAP, mode='full')
    lags = np.arange(-len(norm_HPI) + 1, len(norm_HPI))

    # Normalize the cross-correlation
    norm_factor = np.sqrt(np.sum(np.square(norm_HPI)) * np.sum(np.square(norm_MAP)))
    xcorr_norm = cross_corr / norm_factor

    # Find the index and value of the absolute maximum correlation
    max_corr_index = np.argmax(np.abs(xcorr_norm))
    max_corr_value = xcorr_norm[max_corr_index]
    corresponding_lag = lags[max_corr_index]

    result['Cross correlation HPI & MAP'] = max_corr_value
    result['Time delay HPI & MAP'] = corresponding_lag

    # Initialize the columns with zeros
    data['IOH'] = 0
    data['HPI_alarm'] = 0
    data['MAP_alarm'] = 0

    # Check if MAP is <65 for three consecutive rows (and set IOH=1)
    # Loop through the DataFrame to manually check for sequences of three consecutive values below 65
    for i in range(len(data) - 2):
        # Check if the current value and the next two are below 65
        if data.loc[i, 'MAP_clean'] < 65 and data.loc[i + 1, 'MAP_clean'] < 65 and data.loc[i + 2, 'MAP_clean'] < 65:
            # Mark the current position and the next two positions
            data.loc[i, 'IOH']        = 1
            data.loc[i + 1, 'IOH']    = 1
            data.loc[i + 2, 'IOH']    = 1

    # Check if HPI is >85 for three consecutive rows (and set HPI_alarm=1)
    # Loop through the DataFrame to manually check for sequences of three consecutive values above 85
    for i in range(len(data) - 2):
        # Check if the current value and the next two are is above 85
        if data.loc[i, 'HPI'] > 85 and data.loc[i + 1, 'HPI'] > 85 and data.loc[i + 2, 'HPI'] > 85:
            # Mark the current position and the next two positions
            data.loc[i, 'HPI_alarm']        = 1
            data.loc[i + 1, 'HPI_alarm']    = 1
            data.loc[i + 2, 'HPI_alarm']    = 1

    # Check if MAP < 72 for three consecutive rows (and set MAP_alarm=1)
    # Loop through the DataFrame to manually check for sequences of three consecutive values below 72
    for i in range(len(data) - 2):
        # Check if the current value and the next two are is above 85
        if data.loc[i, 'MAP_clean'] < 72 and data.loc[i + 1, 'MAP_clean'] < 72 and data.loc[i + 2, 'MAP_clean'] < 72:
            # Mark the current position and the next two positions
            data.loc[i, 'MAP_alarm']        = 1
            data.loc[i + 1, 'MAP_alarm']    = 1
            data.loc[i + 2, 'MAP_alarm']    = 1

    minutes_data = len(data)/3
    hypo_samples = sum(data['IOH'])
    hypotension = data['IOH'].eq(1)
    hypo_events = (hypotension&~hypotension.shift(fill_value=False)).cumsum()

    # Calculate the area under the threshold (65mmHg) using the composite trapezoidal rule for every individual episode of hypotension
    aut_vector = data.groupby(hypo_events).apply(lambda x: np.trapz(-x.loc[x['IOH'].eq(1), 'MAP_clean']+65, dx=1/3))

    result['Hypotension samples'] = hypo_samples
    result['Hypotension present'] = hypo_samples > 0
    result['Hypotension duration (minutes)'] = hypo_samples/3
    result['Hypotension duration (%)'] = hypo_samples/len(data)*100
    result['Number of hypotensive events'] = max(hypo_events)
    result['Hypotension AUT'] = aut_vector.sum()
    result['Hypotension TWA'] = aut_vector.sum()/minutes_data

    # Calculate HPI alarm metrics (when alarm for at least 1 minute)
    hpi_samples = sum(data['HPI_alarm'])
    alarming_hpi = data['HPI_alarm'].eq(1)
    hpi_alarms = (alarming_hpi&~alarming_hpi.shift(fill_value=False)).cumsum()

    result['HPI samples'] = hpi_samples
    result['HPI alarm present'] = hpi_samples > 0
    result['HPI alarm duration (minutes)'] = hpi_samples/3
    result['HPI alarm duration (%)'] = hpi_samples/len(data)*100
    result['Number of HPI alarms'] = max(hpi_alarms)

    # Calculate MAP alarm metrics (when alarm for at least 1 minute)
    map_samples = sum(data['MAP_alarm'])
    alarming_map = data['MAP_alarm'].eq(1)
    map_alarms = (alarming_map&~alarming_map.shift(fill_value=False)).cumsum()

    result['MAP alarm samples'] = map_samples
    result['MAP alarm present'] = map_samples > 0
    result['MAP alarm duration (minutes)'] = map_samples/3
    result['MAP alarm duration (%)'] = map_samples/len(data)*100
    result['Number of MAP alarms'] = max(map_alarms)

    # Create empty colums
    data['deltaMAP'] = np.nan
    data['lepMAP'] = np.nan
    data['5min_prediction'] = 0
    data['10min_prediction'] = 0
    data['15min_prediction'] = 0

    # Calculate delta MAP and lineairly extrapolated MAP at 5 minutes
    data['deltaMAP'] = data['MAP_clean'] - data['MAP_clean'].shift(15)
    data['lepMAP'] = data['MAP_clean'] + data['deltaMAP']

    # Create true hypotension prediction with window 1-5 min
    mask = data['IOH'].rolling(window=13, min_periods=3).sum().gt(0).shift(-15).fillna(False)
    data.loc[mask, '5min_prediction'] = 1

    # Create true hypotension prediction with window 1-10 min
    mask = data['IOH'].rolling(window=28, min_periods=3).sum().gt(0).shift(-30).fillna(False)
    data.loc[mask, '10min_prediction'] = 1

    # Create true hypotension prediction with window 1-15 min
    mask = data['IOH'].rolling(window=43, min_periods=3).sum().gt(0).shift(-45).fillna(False)
    data.loc[mask, '15min_prediction'] = 1

    backward_analysis_export = ['HPI_clean','MAP_clean','deltaMAP','lepMAP','5min_prediction', '10min_prediction', '15min_prediction']
    for x in backward_analysis_export: result_vector[x] = data[x].to_list()
    result_vector['Patient ID'] = [identifier] * len(data['HPI_clean'])
    episodes_hpi = data.groupby(hpi_alarms)
    episodes_map = data.groupby(map_alarms)

    for minutes in [5, 10, 15]:
        window_size = minutes * 3 
        # Uncorrected forward analysis for HPI
        tp, fp = 0, 0
        for episode_number, group in episodes_hpi:
            # If the data starts immediately with an alarm, the name of the episode_number is always > 0
            if episode_number == 0:
                pass
            else:
                # Prediction window between 1 minute and window_size
                prediction_window = group[3:window_size + 1]
                # Check if there is hypotension occuring
                hypotension_in_window = prediction_window['IOH'].sum()
                if hypotension_in_window: tp += 1
                else: fp += 1
        result[f'win{minutes}_HPI_alarm_uncorrected_tp'] = tp
        result[f'win{minutes}_HPI_alarm_uncorrected_fp'] = fp

        # Uncorrected forward analysis for MAP
        tp, fp = 0, 0
        for episode_number, group in episodes_map:
            # If the data starts immediately with an alarm, the name of the episode_number is always > 0
            if episode_number == 0:
                pass
            else:
                # Prediction window between 1 minute and window_size
                prediction_window = group[3:window_size + 1]
                # Check if there is hypotension occuring
                hypotension_in_window = prediction_window['IOH'].sum()
                if hypotension_in_window: tp += 1
                else: fp += 1
        result[f'win{minutes}_MAP_alarm_uncorrected_tp'] = tp
        result[f'win{minutes}_MAP_alarm_uncorrected_fp'] = fp

        # Corrected forward analysis for HPI
        # Define functions for two criteria of sudden MAP increases: 
        # - The MAP increased 5 mmHg or more in 20 seconds (1 row in the DataFrame) if MAP < 70
        def condition_5in1(x):
            return(x[0] < 70) and ((x[1] - x[0]) >= 5)
        data['condition_5in1'] = data.MAP_clean.rolling(2).apply(condition_5in1, raw=True)
        # - The MAP increased 8 mmHg or more in two minutes (6 rows in the DataFrame) if MAP < 70
        def condition_8in6(x):
            return(x[0] < 70) and (max(x) - min(x) >= 8) and (x.argmin() < x.argmax())
        data['condition_8in6'] = data.MAP_clean.rolling(6).apply(condition_8in6, raw=True)

        tp, fp = 0, 0
        for episode_number, group in episodes_hpi:
            # If the data starts immediately with an alarm, the name of the episode_number is always > 0
            if episode_number == 0:
                pass
            else:
                # Prediction window between 1 minute and minutes
                prediction_window = group[3:window_size + 1]
                hypotension_in_window = prediction_window['IOH'].sum()
                if hypotension_in_window: tp += 1
                else:
                    if prediction_window['condition_5in1'].sum() == 0 and prediction_window['condition_8in6'].sum() == 0: # There is no correction
                        fp += 1
                    else: # There was a correction so we ignore this episode
                        pass

        result[f'win{minutes}_HPI_alarm_corrected_tp'] = tp
        result[f'win{minutes}_HPI_alarm_corrected_fp'] = fp

        # Corrected forward analysis for MAP
        tp, fp = 0, 0
        for episode_number, group in episodes_map:
            # If the data starts immediately with an alarm, the name of the episode_number is always > 0
            if episode_number == 0:
                pass
            else:
                # Prediction window between 1 minute and minutes
                prediction_window = group[3:window_size + 1]
                hypotension_in_window = prediction_window['IOH'].sum()
                if hypotension_in_window: tp += 1
                else:
                    if prediction_window['condition_5in1'].sum() == 0 and prediction_window['condition_8in6'].sum() == 0: # There is no correction
                        fp += 1
                    else: # There was a correction so we ignore this episode
                        pass

        result[f'win{minutes}_MAP_alarm_corrected_tp'] = tp
        result[f'win{minutes}_MAP_alarm_corrected_fp'] = fp
    return(result, result_vector)

directory = 'input_data'
xls_files = [directory + '/' + f for f in os.listdir(directory) if f.endswith('.xls')]
timepoints  = {'example_patient.xls' : ['01-01-24 08:56:00','01-01-24 16:15:00']}

result = []
result_vector = []

for filename in xls_files:
    identifier = os.path.split(filename)[1]
    columns = pd.read_excel(filename).columns
    if 'Date' in columns:
        raw_data = pd.read_excel(filename, skiprows=[1])
        raw_data['Time'] = pd.to_datetime(raw_data['Time'], format='%H:%M:%S')
        raw_data['datetime'] = pd.to_datetime(raw_data['Date'].dt.date.astype(str) + ' ' + raw_data['Time'].dt.time.astype(str))
        tsdt = datetime.strptime(timepoints[identifier][0], '%d-%m-%y %H:%M:%S')
        tedt = datetime.strptime(timepoints[identifier][1], '%d-%m-%y %H:%M:%S')
    elif 'Datum' in columns: # New datatype
        datetime_format = '%d.%m.%Y %H:%M:%S'
        parse_dates = {'datetime' : ['Datum','Time']}
        raw_data = pd.read_excel(filename, parse_dates = parse_dates, date_format=datetime_format, skiprows=[1])
        tsdt = datetime.strptime(timepoints[identifier][0], '%d-%m-%y %H:%M:%S')
        tedt = datetime.strptime(timepoints[identifier][1], '%d-%m-%y %H:%M:%S')

    r, rv = analyse_data(raw_data, identifier, tsdt, tedt)
    result.append(r)
    result_vector.append(rv)

result_df = pd.DataFrame(result)
result_vector_df = pd.concat([pd.DataFrame(x) for x in result_vector], ignore_index=True)
result_df.to_excel('result.xlsx')
result_vector_df.to_pickle('result_vector.pickle')

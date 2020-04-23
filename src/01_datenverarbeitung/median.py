import numpy as np

from iul.iul import IUL

iul = IUL(False)
exercise_id = "median"
data = iul.get_data(exercise_id)

data_session = data["session"]
data_signals = data["signals"]
signal_results = []

for data_signal in data_signals:
    signal_id = data_signal["id"]
    signal_values = data_signal["values"]
    signal_values = np.sort(signal_values)
    values_len = len(signal_values)

    median = 0.0
    if values_len % 2 == 0:
        middle = int(values_len / 2)
        median = (signal_values[middle - 1] + signal_values[middle]) / 2
    else:
        middle = int((values_len - 1) / 2)
        median = signal_values[middle]

    signal_result = {
        'id': signal_id,
        'median': median
    }
    signal_results.append(signal_result)
result = {
    'session': data_session,
    'results': signal_results
}

iul.post_result(exercise_id, result)

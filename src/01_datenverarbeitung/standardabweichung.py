import math

import numpy as np

from iul.iul import IUL

iul = IUL(False)
exercise_id = "standard_deviation"
data = iul.get_data(exercise_id)

data_session = data["session"]
data_signals = data["signals"]
signal_results = []

for data_signal in data_signals:
    signal_id = data_signal["id"]
    signal_values = data_signal["values"]
    mean = 0.0
    for signal_value in signal_values:
        mean += signal_value
    mean /= len(signal_values)

    standard_deviation = 0.0
    for signal_value in signal_values:
        standard_deviation += math.pow(signal_value - mean, 2)
    standard_deviation /= len(signal_values)
    standard_deviation = math.sqrt(standard_deviation)
    signal_result = {
        'id': signal_id,
        'standard_deviation': standard_deviation
    }
    print("Numpy: {}\nOwn:   {}\n".format(np.std(signal_values), standard_deviation))
    signal_results.append(signal_result)
result = {
    'session': data_session,
    'results': signal_results
}
iul.post_result(exercise_id, result)

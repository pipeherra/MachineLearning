from iul.iul import IUL

iul = IUL(False)
exercise_id = "moving_average"
data = iul.get_data(exercise_id)

data_session = data["session"]
data_signals = data["signals"]
signal_results = []

for data_signal in data_signals:
    signal_id = data_signal["id"]
    signal_values = data_signal["values"]
    signal_window = data_signal["window"]
    if signal_window % 2 == 0:
        print("Signal-Window is even: {}".format(signal_window))
        continue
    signal_averages = []
    for i in range(len(signal_values) - signal_window + 1):
        signal_average = 0.0
        for j in range(signal_window):
            signal_average += signal_values[i+j]
        signal_average /= signal_window
        signal_averages.append(signal_average)
    signal_result = {
        'id': signal_id,
        'values': signal_averages
    }
    signal_results.append(signal_result)

result = {
    'session': data_session,
    'results': signal_results
}

iul.post_result(exercise_id, result)


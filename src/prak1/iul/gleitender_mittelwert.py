from src.iul.iul import IUL
from stats.statistic_old import get_moving_average

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
    signal_result = {
        'id': signal_id,
        'values': get_moving_average(signal_values, signal_window).tolist()
    }
    signal_results.append(signal_result)

result = {
    'session': data_session,
    'results': signal_results
}

iul.post_result(exercise_id, result)

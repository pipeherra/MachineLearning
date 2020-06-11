from src.iul.iul import IUL
from stats.statistic import Statistic

iul = IUL(False)
exercise_id = "median"
data = iul.get_data(exercise_id)

data_session = data["session"]
data_signals = data["signals"]
signal_results = []

for data_signal in data_signals:
    signal_id = data_signal["id"]
    signal_values = data_signal["values"]
    signal_result = {
        'id': signal_id,
        'median': Statistic.get_median(signal_values)
    }
    signal_results.append(signal_result)
result = {
    'session': data_session,
    'results': signal_results
}

iul.post_result(exercise_id, result)

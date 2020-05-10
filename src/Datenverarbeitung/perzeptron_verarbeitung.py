import math

from src.Datenverarbeitung.perceptron import Perceptron
from src.iul.iul import IUL
import numpy as np

iul = IUL(False)
exercise_id = "perceptron_processing"
request_data = iul.get_data(exercise_id)

request_data_session = request_data["session"]
attribute_count = request_data["attribute-count"]
data_array = request_data["data"]

results = []
for data in data_array:
    data_id = data["id"]
    data_weights = data["weights"]
    data_threshold = data_weights.pop(0)
    data_inputs = data["input"]

    data_sum = data_threshold
    for (data_weight, data_input) in zip(data_weights, data_inputs):
        data_sum += data_weight * data_input
    data_value = math.tanh(data_sum)
    data_value_norm = (data_value * 0.5) + 0.5
    print("ID: {}, Threshold: {}, Weights-Len: {}, Inputs-Len:{}, Data-Sum: {}, Data-Value: {}, Data-Value-Norm: {}"
          .format(data_id, data_threshold, len(data_weights), len(data_inputs), data_sum, data_value, data_value_norm))

    data_result = {
        'id': data_id,
        'value': data_value_norm
    }
    results.append(data_result)

result = {
    'session': request_data_session,
    'results': results
}

iul.post_result(exercise_id, result)


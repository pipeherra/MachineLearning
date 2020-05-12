from algorithmen.perceptron import Perceptron
from src.iul.iul import IUL

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
    data_inputs = data["input"]
    data_inputs.insert(0, 1.0)
    perceptron = Perceptron(data_weights)
    data_value = perceptron.predict_with_normalized_tan(data_inputs)

    data_result = {
        'id': data_id,
        'value': data_value
    }
    results.append(data_result)

result = {
    'session': request_data_session,
    'results': results
}

iul.post_result(exercise_id, result)


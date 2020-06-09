from algorithms.classification import Classification
from algorithms.data_point import DataPoint
from algorithms.perceptron import Perceptron
from algorithms.transfers.tanh import Tanh
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

    perceptron = Perceptron(Classification.get_true_false(), data_weights, Tanh())
    prediction = perceptron.predict_data(DataPoint(data_inputs, Classification(0.0, "False")))

    data_result = {
        'id': data_id,
        'value': prediction.value
    }
    results.append(data_result)

result = {
    'session': request_data_session,
    'results': results
}

iul.post_result(exercise_id, result)

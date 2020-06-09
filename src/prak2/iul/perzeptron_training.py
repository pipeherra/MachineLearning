from algorithms.classification import Classification
from algorithms.data_point import DataPoint
from algorithms.perceptron import Perceptron
from algorithms.transfers.tanh import Tanh
from src.iul.iul import IUL

iul = IUL(False)
exercise_id = "perceptron_training"
request_data = iul.get_data(exercise_id)

request_data_session = request_data["session"]
attribute_count = request_data["attribute-count"]
learning_rate = request_data["learning-rate"]
initial_weights = request_data["initial-weights"]
training_data_array = request_data["training-data"]

perceptron = Perceptron(Classification.get_true_false(), initial_weights, Tanh(), learning_rate=learning_rate)

results = []
for training_data in training_data_array:
    data_id = training_data["id"]
    data_inputs = training_data["input"]
    data_inputs.insert(0, 1.0)
    data_expected = training_data['class']
    result_weights = perceptron.update_weights(DataPoint(data_inputs, Classification(data_expected, "unknown")))
    data_result = {
        'id': data_id,
        'weights': result_weights.copy()
    }
    results.append(data_result)

result = {
    'session': request_data_session,
    'results': results
}

iul.post_result(exercise_id, result)

from src.Datenverarbeitung.perceptron import Perceptron
from src.iul.iul import IUL

iul = IUL(False)
exercise_id = "perceptron_training"
request_data = iul.get_data(exercise_id)

request_data_session = request_data["session"]
attribute_count = request_data["attribute-count"]
learning_rate = request_data["learning-rate"]
data_weights = request_data["initial-weights"]
data_array = request_data["training-data"]

perceptron = Perceptron()

threshold = data_weights.pop(0)

print(data_weights)
weights = data_weights[1]
print(weights)

results = []
for data in data_array:
    data_id = data["id"]
    data_inputs = data["input"]
    data_class = data['class']
    data_value = perceptron.train(data_weights, data_inputs, data_class, learning_rate, threshold)
    data_weights = data_value
    data_result = {
        'id': data_id,
        'weights': data_value
    }
    results.append(data_result)

result = {
    'session': request_data_session,
    'results': results
}

iul.post_result(exercise_id, result)


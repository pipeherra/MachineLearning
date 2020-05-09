from src.Datenverarbeitung.perceptron import Perceptron
from src.iul.iul import IUL
import numpy as np


iul = IUL(False)
exercise_id = "perceptron_processing"
daten = iul.get_data(exercise_id)

data_session = daten["session"]
data = daten["data"]
data_attribute_count = daten["attribute-count"]
results = []
perceptron = Perceptron(maxEpochs=10000, learning_rate=0.1, features=data_attribute_count)

for dat in data:
    data_id = dat["id"]
    data_weights = dat["weights"]
    data_input = dat["input"]
    data_result = {
        'id': data_id,
        'value': perceptron.guess(data_input).tolist()
    }
    results.append(data_result)

result = {
    'session': data_session,
    'results': results
}

iul.post_result(exercise_id, result)


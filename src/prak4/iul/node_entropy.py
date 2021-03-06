from typing import List

from misc.classification import Classification
from misc.data_point import DataPoint
from algorithms.decision_tree.leaf import Leaf
from src.iul.iul import IUL

iul = IUL(False)
exercise_id = "node_entropy"
request_data = iul.get_data(exercise_id)

test_id = request_data['session']
data_array = request_data['data']

classifications: List[Classification] = []
inputs: List[DataPoint] = []

for data in data_array:
    clazz = data['class']
    current_class = Classification(clazz, "unknown")
    if current_class not in classifications:
        classifications.append(current_class)
    inputs.append(DataPoint(data['input'], current_class))

leaf = Leaf(inputs)
entropy = leaf.get_entropy()

print(entropy)

result_data = {'session': test_id,
               'entropy': entropy}

response = iul.post_result(exercise_id, result_data)

# Antwort der TEstaufgabe auf IUL
print("Auswertung:")
for c in response['classification']:
    print("Klasse: " + str(c['class']) + " | Wahrscheinlichkeit: " + str(c['probability']))

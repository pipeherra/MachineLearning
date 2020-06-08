from src.iul.iul import IUL
import math
import numpy as np

iul = IUL(False)
exercise_id = "node_entropy"
request_data = iul.get_data(exercise_id)

test_id = request_data['session']
classifications = request_data['data']

instances_total = 0 #len(classifications)
instances_per_class = dict()

for classification in classifications:
    clazz = classification['class']
    features = classification['input']
    instances_per_class[clazz] = instances_per_class.get(clazz, 0) + 1
    instances_total += 1

entropy = 0.0
for clazz in instances_per_class.keys():
    probability = instances_per_class[clazz] / instances_total
    if probability != 0:
        temp_entropy = -probability * np.log(probability)
        entropy += temp_entropy
    else:
        print("probability wrong: {}".format(probability))

print(instances_total)
print(instances_per_class)
print(entropy)

result_data = {'session': test_id,
               'entropy': entropy}

response = iul.post_result(exercise_id, result_data)

# Antwort der TEstaufgabe auf IUL
print("Auswertung:")
for c in response['classification']:
    print("Klasse: " + str(c['class']) + " | Wahrscheinlichkeit: " + str(c['probability']))

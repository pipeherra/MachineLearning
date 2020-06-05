from src.iul.iul import IUL
import math

iul = IUL(False)
exercise_id = "node_entropy"
request_data = iul.get_data(exercise_id)

test_id = request_data['session']
classifications = request_data['data']

instances_total = len(classifications)
instances_per_class = dict()

for classification in classifications:
    clazz = classification['class']
    features = classification['input']
    instances_per_class[clazz] = instances_per_class.get(clazz, 0) + 1

probabilities = dict()
for clazz in instances_per_class.keys():
    probabilities[clazz] = instances_per_class[clazz] / instances_total

entropy = 0.0
for probability in probabilities.values():
    if 0.0 < probability < 1.0:
        entropy -= probability * math.log(probability)

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

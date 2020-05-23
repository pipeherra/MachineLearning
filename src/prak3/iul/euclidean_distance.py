from src.iul.iul import IUL
from src.algorithmen.metric import get_distance

iul = IUL(False)
exercise_id = "euclidean_distance"
request_data = iul.get_data(exercise_id)

test_id = request_data['session']
metric = request_data['metric']
routes = request_data['routes']

print(metric)
print(routes)

result = []
for x in routes:
    result += [{'id': x['id'], 'distance': get_distance(x['from'], x['to'], metric)}]

result_data = {'session': test_id,
               'results': result}

response = iul.post_result(exercise_id, result_data)

# Antwort der TEstaufgabe auf IUL
print("Auswertung:")
for c in response['classification']:
    print("Klasse: " + str(c['class']) + " | Wahrscheinlichkeit: " + str(c['probability']))

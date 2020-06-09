from algorithms.classification import Classification
from algorithms.data_point import DataPoint
from algorithms.metrics.chessboard import Chessboard
from algorithms.metrics.euclidean import Euclidean
from algorithms.metrics.manhattan import Manhattan
from src.iul.iul import IUL

iul = IUL(False)
exercise_id = "chessboard_distance"
request_data = iul.get_data(exercise_id)

test_id = request_data['session']
metric_str = request_data['metric']
routes = request_data['routes']

if metric_str == 'chessboard':
    metric = Chessboard()
elif metric_str == 'euclidean':
    metric = Euclidean()
elif metric_str == 'manhattan':
    metric = Manhattan()
else:
    raise AttributeError("Unknown metric: {}".format(metric_str))
print(metric_str)
print(routes)

classification_from = Classification(0.0, "From")
classification_to = Classification(1.0, "To")

result = []
for x in routes:
    distance = metric.get_distance(DataPoint(x['from'], classification_from), DataPoint(x['to'], classification_to))
    result += [{'id': x['id'], 'distance': distance}]

# print(result)

result_data = {'session': test_id,
               'results': result}

response = iul.post_result(exercise_id, result_data)

# Antwort der TEstaufgabe auf IUL
print("Auswertung:")
for c in response['classification']:
    print("Klasse: " + str(c['class']) + " | Wahrscheinlichkeit: " + str(c['probability']))

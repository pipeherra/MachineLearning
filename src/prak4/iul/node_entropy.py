from src.iul.iul import IUL

iul = IUL(True)
exercise_id = "node_entropy"
request_data = iul.get_data(exercise_id)

test_id = request_data['session']
data = request_data['data']

result_data = {'session': test_id,
               'entropy': 0.0}

response = iul.post_result(exercise_id, result_data)

# Antwort der TEstaufgabe auf IUL
print("Auswertung:")
for c in response['classification']:
    print("Klasse: " + str(c['class']) + " | Wahrscheinlichkeit: " + str(c['probability']))

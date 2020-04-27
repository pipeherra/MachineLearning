from src.iul.iul import IUL

iul = IUL()
exercise_id = "echo"
data = iul.get_data(exercise_id)

data_session = data['session']
data_value = data['value']

result = {
    'session': data_session,
    'result': data_value,
}
iul.post_result(exercise_id, result)

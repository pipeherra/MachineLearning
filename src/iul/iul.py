from pprint import pprint

import requests
import json

from src.iul.config import Config


class IUL:
    def __init__(self, debug=True, config=Config()):
        self.config = config
        self.debug = debug

    def get_data(self, excercise_id):
        data_url = self.config.host + "/exercises/" + excercise_id + "/data"
        try:
            response = requests.get(data_url, headers={'Authorization': 'Token ' + self.config.token})
            if response.status_code != 200:
                print("Could not get data from IUL-server")
                exit(1)
            json_data = json.loads(response.text)
            if self.debug:
                print("\nReceived exercise from server\n-------------------------------")
                pprint(json_data)
                print("-------------------------------\n\n")
            return json_data
        except Exception as ex:
            print("Exception: Could not get data from IUL-Server")
            print(ex)
            exit(1)
        return ""

    def post_result(self, excercise_id, result):
        if self.debug:
            print("Calculated result\n-------------------------------")
            pprint(result)
            print("-------------------------------\n\n")
        result_url = self.config.host + "/exercises/" + excercise_id + "/result"
        try:
            response = requests.post(result_url, json=result, headers={'Authorization': 'Token ' + self.config.token, 'content-type': 'application/json'})
            if response.status_code != 200:
                print("Could not post result to IUL-server")
                exit(1)
            json_classification = json.loads(response.text)
            if self.debug:
                print("\nReceived classification from server\n-------------------------------")
                pprint(json_classification)
                print("-------------------------------\n")
            return json_classification
        except Exception as ex:
            print("Exception: Could not post result to IUL-Server")
            print(ex)
            exit(1)
        return ""

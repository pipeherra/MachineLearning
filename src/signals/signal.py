from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from misc.classification import Classification
from misc.sensor import Sensor


class Signal:
    sensors: List[Sensor]
    max_timestamp: int
    min_timestamp: int

    def __init__(self, path: str, classification: Classification, start: int, end: int):
        """

        :type end: end of sequence in ms (max_timestamp - end)
        :type start: start of sequence in ms (min_timestamp + start)
        """
        data = pd.read_csv(path)
        self.classification = classification
        timestamps = data.Timestamp
        max_timestamp = max(timestamps)
        min_timestamp = min(timestamps)
        data = data[((max_timestamp - end) > data.Timestamp) & ((min_timestamp + start) < data.Timestamp)]
        timestamps = data.Timestamp
        self.min_timestamp = min(timestamps)

        timestamps_normalized = []
        for timestamp in timestamps:
            timestamps_normalized.append(timestamp - self.min_timestamp)
        data['Timestamp_normalized'] = timestamps_normalized
        self.data = data
        self.data_len = max(timestamps_normalized)

    def plot(self, column: str, label: str = ""):
        if label == "":
            label = self.classification.name
        plt.scatter(self.data['Timestamp_normalized'], self.data[column], label=label)



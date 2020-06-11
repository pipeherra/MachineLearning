import pandas as pd


def get_moving_average(signals, window):
    averages = []
    indexes = []
    if window % 2 == 0:
        print("Signal-Window is even: {}".format(window))
        return averages
    index = int((window - 1) / 2)
    for i in range(len(signals) - window + 1):
        average = 0.0
        for j in range(window):
            average += signals[i + j]
        average /= window
        averages.append(average)
        indexes.append(index)
        index = index + 1
    return pd.Series(averages, indexes)

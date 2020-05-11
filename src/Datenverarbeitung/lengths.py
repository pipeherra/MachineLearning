import glob
import os

import pandas as pd

from signals.statistics import Statistics

for file in glob.glob("../../data/17SoSe/2017_Gruppe6_Appelfeller-Krupa/*.csv"):
    data = pd.read_csv(file)
    timestamps = data['Timestamp']
    data_len = Statistics.get_len(timestamps)
    print("{}: Data-Count: {}, Duration(Timestamps): {}, Duration (sec): {}"
          .format(os.path.basename(file), len(timestamps), data_len, data_len/1000.0))

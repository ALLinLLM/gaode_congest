import pandas as pd
import numpy as np
data = pd.read_pickle("/workdir/congest/result/detection.pkl")
print(data[['id', 'person_num', "nonvehicle_num", "vehicle_num"]].groupby(['id']).agg(np.mean, np.std, np.max, np.min))
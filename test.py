import pandas as pd
import numpy as np

df = pd.read_csv('plankton.csv')
df.drop_duplicates(subset='im_name', inplace=True, keep=False)

print(np.unique(df.label))

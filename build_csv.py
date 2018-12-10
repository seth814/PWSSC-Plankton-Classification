import pandas as pd
import os
from glob import glob
import numpy as np
import pickle

# uncomment mod_labels to build multi classification
mod_labels = {}

mod_labels = {'acartia': [0, 9],
              'calanus': [5, 9],
              'metridia': [17, 9],
              'neocalanus': [19, 9],
              'oithona': [21, 9],
              'pseudocalanus': [27, 9]
              }


class_map = {}
im_names = []
labels = []
cwd = os.getcwd()
train_path = os.path.join(cwd, 'multi_padded')
classes = os.listdir(train_path)
for i, c in enumerate(classes):
    class_map[i] = c
    if c.lower() in mod_labels.keys():
        i = np.array(mod_labels[c.lower()], dtype=int)
    else:
        i = np.array(i, dtype=int)
    class_dir = os.path.join(train_path, c)
    os.chdir(class_dir)
    images = glob('*.png')
    for im in images:
        im_names.append(im)
        labels.append(i)

d = {'im_name': im_names, 'label': labels}
df = pd.DataFrame(data=d)
os.chdir(cwd)

df.to_csv('plankton_multi.csv', index=False)

with open('class_map_multi.pickle', 'wb') as handle:
    pickle.dump(class_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

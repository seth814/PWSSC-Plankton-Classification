import pandas as pd
import os
import numpy as np
import pickle

# uncomment mod_labels to build multi classification
mod_labels = {}
'''
mod_labels = {'acartia': [0, 9, 10]}
'''
class_map = {}
im_names = []
labels = []
train_path = os.path.join(os.getcwd(), 'data')
exists = os.path.join(os.getcwd(), 'pad')
if os.path.isdir(exists) is False:
    os.mkdir(exists)
classes = os.listdir(train_path)
for i, c in enumerate(classes):
    class_map[i] = c
    if c.lower() in mod_labels.keys():
        i = np.array(mod_labels[c.lower()], dtype=int)
    else:
        i = np.array(i, dtype=int)
    class_dir = os.path.join(train_path, c)
    images = os.listdir(class_dir)
    for im in images:
        im_names.append(im)
        labels.append(i)

print(class_map)

d = {'im_name': im_names, 'label': labels}
df = pd.DataFrame(data=d)

df.to_csv('plankton.csv', index=False)

with open('class_map.pickle', 'wb') as handle:
    pickle.dump(class_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

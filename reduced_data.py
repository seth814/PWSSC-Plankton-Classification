import numpy as np
import pandas as pd
import os
from glob import glob

frames = []
cwd = os.getcwd()
train_path = os.path.join(cwd, 'data')
classes = os.listdir(train_path)
for i, c in enumerate(classes):
    class_dir = os.path.join(train_path, c)
    os.chdir(class_dir)
    images = glob('*.png')
    for im in images:
        frames.append((im, i))

frames = np.array(frames)
d = {'im_name': frames[:,0], 'label': frames[:,1]}
df = pd.DataFrame(data=d)
os.chdir(cwd)
df.to_csv('plankton.csv', index=False)

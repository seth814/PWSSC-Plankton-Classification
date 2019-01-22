import numpy as np
import pandas as pd
from collections import defaultdict
import pickle

def find_pairs(repeats):
    pairs = defaultdict(list)
    for name, c in zip(df.im_name, df.label):
        if name in repeats:
            print(name, c)
            pairs[name].append(c)
    return pairs

df = pd.read_csv('plankton.csv')
print(len(df.im_name))
print(len(np.unique(df.im_name)))

repeats = []
names = defaultdict(int)
for name, c in zip(df.im_name, df.label):
    if names[name] == 1:
        print('Repeat found: {}, class: {}'.format(name, c))
        repeats.append(name)
    names[name] = 1

pairs = find_pairs(repeats)
print(pairs.items())
print('{} repeated images found'.format(len(pairs.keys())))


with open('repeats.p', 'wb') as handle:
    pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

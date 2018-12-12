import os
import pandas as pd
import pickle
import cv2
from sklearn.preprocessing import MinMaxScaler
from segmentation import Segmentation

'''
This file will iterate through every image in the data directory.
It builds padded and features so a segmentation object does not need to be built during modeling.

creates:
 - features.pickle: a dictionary where im_name is keys and feats are values ( O(1) time lookup )
 - extracted_features.csv: a csv file of features
 - normalizer.pickle: a MinMaxScaler object to normalize features at run time
 - abc.png: an image padded after segmenation

extracted_features.csv is also built here so run this if you change the data.
The normalizer will also be built using extracted features.
Duplicate image names will be removed.
'''

with open('class_map.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

df = pd.read_csv('plankton.csv')
df.drop_duplicates(subset='im_name', inplace=True, keep=False)

feats = {}
feat_list = []

data_path = os.path.join(os.getcwd(), 'data')
pad_path = os.path.join(os.getcwd(), 'pad')
classes = os.listdir(data_path)

for c in classes:
    im_dir = os.path.join(data_path, c)
    exists = os.path.join(pad_path, c)
    if os.path.isdir(exists) is False:
        os.mkdir(exists)
    images = os.listdir(im_dir)
    for im_name in images:
        im_path = os.path.join(im_dir, im_name)
        im_pad_path = os.path.join(exists, im_name)

        plankton = Segmentation(im_path, target_shape=(75, 75, 3))
        plankton.segment()
        padded = plankton.get_padded()
        padded = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
        cv2.imwrite(im_pad_path, padded)
        feats[im_name] = plankton.get_features()
        feat_list.append(plankton.get_features())

with open('features.pickle', 'wb') as handle:
    pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

df_feats = pd.DataFrame(data=feat_list, columns=plankton.get_columns())
df_feats.to_csv('extracted_features.csv', index=False)

mms = MinMaxScaler()
mms.fit(df_feats.values)

with open('normalizer.pickle', 'wb') as handle:
    pickle.dump(mms, handle, protocol=pickle.HIGHEST_PROTOCOL)

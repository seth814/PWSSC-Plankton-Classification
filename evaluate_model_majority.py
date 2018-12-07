import pandas as pd
import pickle
from segmentation import Segmentation
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from keras.models import load_model
from augment import seq
from collections import Counter

with open('class_map.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

with open('normalizer.pickle', 'rb') as handle:
    mms = pickle.load(handle)

df = pd.read_csv('plankton.csv')
df.drop_duplicates(subset='im_name', inplace=True)
df = df[df.label != 36]

model = load_model('./models/inception_v3_3k_gen_static_val.model')
y_pred = []

input_shape = (75, 75, 3)
feat_shape = (23,)

data_path = os.path.join(os.getcwd(), 'data')

for i, (im_name, label) in enumerate(zip(df.im_name, df.label)):
    im_dir = os.path.join(data_path, class_map[label])
    im_path = os.path.join(im_dir, im_name)
    plankton = Segmentation(im_path, target_shape=input_shape)
    plankton.segment()
    padded = plankton.get_padded()

    X_img = np.empty((5, input_shape[0], input_shape[1], input_shape[2]), dtype=np.uint8)
    X_feat = np.empty((5, feat_shape[0]))

    for i in range(5):
        aug = seq.augment_image(padded)
        X_img[i,] = preprocess_input(aug)
        X_feat[i,] = plankton.get_features()

    X_feat = mms.transform(X_feat)
    y_hat = model.predict([X_img, X_feat])
    votes = [np.argmax(y) for y in y_hat]
    counter = Counter(votes)
    y_hat, num = counter.most_common()[0]
    if y_hat == 36 and len(counter) > 1:
        y_hat, num = counter.most_common()[1]
    elif y_hat == 36:
        y_hat = np.random.randint(36)
    y_pred.append(y_hat)

d = {'y_true': df.label, 'y_pred': y_pred}
df_results = pd.DataFrame(data=d)
df_results.to_csv('./model_results/3k_gen_majority', index=False)

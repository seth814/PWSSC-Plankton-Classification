import pandas as pd
import pickle
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from keras.models import load_model
import cv2
from custom_metrics import f1
from sklearn.manifold import TSNE

def eval_model(model, prob_threshold=0.95):
    y_pred = []
    for i in range(X_img.shape[0]):
        x_img = X_img[i].reshape(1, input_shape[0], input_shape[1], input_shape[2])
        x_feat = X_feat[i].reshape(1, feat_shape[0])
        y_hat = model.predict([x_img, x_feat])
        y_pred.append(y_hat.flatten())
    return np.array(y_pred)


with open('class_map.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

with open('normalizer.pickle', 'rb') as handle:
    mms = pickle.load(handle)

with open('features.pickle', 'rb') as handle:
    features = pickle.load(handle)

df = pd.read_csv('plankton.csv')
df.drop_duplicates(subset='im_name', inplace=True, keep=False)

input_shape = (75, 75, 3)
feat_shape = (16,)

X_img = np.empty((df.shape[0], input_shape[0], input_shape[1], input_shape[2]), dtype=np.uint8)
X_feat = np.empty((df.shape[0], feat_shape[0]))
y = []

data_path = os.path.join(os.getcwd(), 'pad')
classes = os.listdir(data_path)

hash = dict(zip(df.im_name, df.label.values))
i = 0

for z, c in enumerate(classes):
    im_dir = os.path.join(data_path, c)
    images = os.listdir(im_dir)
    images = np.array([i for i in images if i in hash.keys()])
    for im in images:
        im_path = os.path.join(im_dir, im)
        padded = cv2.imread(im_path, -1)
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        X_img[i,] = preprocess_input(padded)
        _, im_name = os.path.split(im_path)
        X_feat[i,] = features[im_name]
        y.append(z)
        i += 1

X_feat = mms.transform(X_feat)

X = [X_img, X_feat]
y = np.array(y).reshape((len(y), 1))

model = load_model('./models/inception_v3_multi.model', custom_objects={'f1': f1})
y_pred = eval_model(model)
print(y_pred.shape, y.shape)

df_result = pd.DataFrame(y_pred)
df_result.to_csv('./model_results/multi_prob.csv', index=False)

tsne = TSNE(n_components=2)
X = tsne.fit_transform(y_pred)
X = np.concatenate((X, y), axis=1)
df_result = pd.DataFrame(data=X)
df_result.to_csv('./model_results/multi_tsne.csv', index=False)

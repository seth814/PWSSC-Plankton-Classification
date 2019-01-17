import pandas as pd
import pickle
import numpy as np
import os
from keras.models import load_model
from custom_metrics import f1
from sklearn.manifold import TSNE
import re
from sklearn.metrics import f1_score
from PIL import Image

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def eval_model(model, prob_threshold=0.95):
    y_probs = []
    y_pred = []
    index = []
    for i in range(X_img.shape[0]):
        x_img = X_img[i].reshape(1, input_shape[0], input_shape[1], input_shape[2])
        x_feat = X_feat[i].reshape(1, feat_shape[0])
        y_hat = model.predict([x_img, x_feat])
        y_probs.append(y_hat.flatten())

        valid = False
        container = np.zeros(shape=(1,n_classes), dtype=int)
        for z, p in enumerate(y_hat.flatten()):
            if p > prob_threshold:
                container[0][z] = 1
                valid = True
        if valid:
            index.append(i)
            y_pred.append(container)

    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[2])

    return np.array(y_probs), (np.array(index), y_pred)


def encode_labels(labels):
    encoded = np.zeros(shape=(1, n_classes), dtype=float)
    labels = [int(s) for s in re.findall(r'\b\d+\b', labels)]
    for l in labels:
        encoded[0][l] = 1.0

    return encoded


with open('class_map.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

with open('normalizer.pickle', 'rb') as handle:
    mms = pickle.load(handle)

with open('features.pickle', 'rb') as handle:
    features = pickle.load(handle)

df = pd.read_csv('plankton.csv')
df.drop_duplicates(subset='im_name', inplace=True, keep=False)
df.reset_index(inplace=True)

n_classes = 34
input_shape = (75, 75, 3)
feat_shape = (16,)

X_img = np.empty((df.shape[0], input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32)
X_feat = np.empty((df.shape[0], feat_shape[0]))
y = []
y_hot = []

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
        padded = Image.open(im_path)
        padded = np.array(padded, dtype=np.uint8)
        X_img[i,] = preprocess_input(np.array(padded, dtype=np.float32))
        _, im_name = os.path.split(im_path)
        X_feat[i,] = features[im_name]
        y.append(z)
        y_hot.append(encode_labels(hash[im]))
        i += 1

X_feat = mms.transform(X_feat)

X = [X_img, X_feat]
y = np.array(y)
y = y.reshape((y.shape[0], 1))
y_hot = np.array(y_hot)
y_hot = y_hot.reshape(y_hot.shape[0], y_hot.shape[2])

model = load_model('./models/test.model', custom_objects={'f1': f1})
y_prob, hot_encoded = eval_model(model, prob_threshold=0.95)
(index, y_pred) = hot_encoded

f1 = f1_score(y_true=y_hot[index], y_pred=y_pred, average='weighted')
percent = str(round(index.shape[0] / df.shape[0] * 100, 3))
print('Classified {}% of the data using a threshold of 95%'.format(percent))
print('Weighted F1-Score: {:4.4f}'.format(f1))

df_result = pd.DataFrame(y_prob)
df_result.to_csv('./model_results/test_prob.csv', index=False)

# performs tsne decomposition (warning: tsne compares elements quadratically)
# comment out if data is too large
'''
tsne = TSNE(n_components=2)
X = tsne.fit_transform(y_prob)
X = np.concatenate((X, y), axis=1)
df_result = pd.DataFrame(data=X)
df_result.to_csv('./model_results/test_tsne.csv', index=False)
'''

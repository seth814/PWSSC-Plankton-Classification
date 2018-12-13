import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import cv2

def eval_model(model):
    y_pred = []
    for i in range(X_img.shape[0]):
        x_img = X_img[i].reshape(1, input_shape[0], input_shape[1], input_shape[2])
        x_feat = X_feat[i].reshape(1, feat_shape[0])
        y_hat = model.predict([x_img, x_feat])
        y_pred.append(np.argmax(y_hat))
    return y_pred

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

data_path = os.path.join(os.getcwd(), 'pad')

for i, (im_name, label) in enumerate(zip(df.im_name, df.label)):
    im_dir = os.path.join(data_path, class_map[label])
    im_path = os.path.join(im_dir, im_name)
    padded = cv2.imread(im_path, -1)
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    X_img[i,] = preprocess_input(padded)
    _, im_name = os.path.split(im_path)
    X_feat[i,] = features[im_name]

X_feat = mms.transform(X_feat)

X = [X_img, X_feat]

model = load_model('./models/inception_v3.model')
y_pred = eval_model(model)

d = {'y_true': df.label, 'y_pred': y_pred}
df_results = pd.DataFrame(data=d)
df_results.to_csv('./model_results/inception_v3.csv', index=False)

acc = str(round(accuracy_score(df.label, y_pred), 4))
f1 = str(round(f1_score(df.label, y_pred, average='weighted'), 4))
conf_mat = confusion_matrix(df.label, y_pred)
conf_mat = conf_mat.astype(dtype=np.float16)
for row in range(conf_mat.shape[0]):
    conf_mat[row,:] = conf_mat[row,:] / sum(conf_mat[row,:])

plt.title('Confusion Matrix')
sns.heatmap(conf_mat, cmap='hot', square=True, xticklabels=False, yticklabels=False)
plt.show()

print('Accuracy: {}'.format(acc))
print('F1 Score: {}'.format(f1))

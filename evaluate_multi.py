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
from custom_metrics import f1

def drop_classes(df, labels, class_map):

    print(df.shape)
    for i in labels:
        print('Dropping {}'.format(class_map[i]))
        df = df[df.label != i]
        class_map.pop(i)
    print(df.shape)

    new_class_map = {}
    for current, new in zip(np.unique(df.label), range(len(class_map))):
        df.loc[df.label == current, 'label'] = new
        new_class_map[new] = class_map[current]

    with open('dropped_4_class_map.pickle', 'wb') as handle:
        pickle.dump(new_class_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return df, new_class_map

def eval_model(model, prob_threshold=0.95):
    y_pred = []
    for i in range(X_img.shape[0]):
        encoded = np.zeros(shape=(1,len(class_map)))
        x_img = X_img[i].reshape(1, input_shape[0], input_shape[1], input_shape[2])
        x_feat = X_feat[i].reshape(1, feat_shape[0])
        y_hat = model.predict([x_img, x_feat])
        for i, y in enumerate(y_hat.flatten()):
            encoded[0][i] = 1 if y > prob_threshold else 0
        y_pred.append(encoded)
        print(encoded)
    return y_pred

with open('class_map_multi.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

with open('normalizer.pickle', 'rb') as handle:
    mms = pickle.load(handle)

with open('features_multi.pickle', 'rb') as handle:
    features = pickle.load(handle)

df = pd.read_csv('plankton_multi.csv')
df.drop_duplicates(subset='im_name', inplace=True, keep=False)

input_shape = (75, 75, 3)
feat_shape = (16,)

X_img = np.empty((df.shape[0], input_shape[0], input_shape[1], input_shape[2]), dtype=np.uint8)
X_feat = np.empty((df.shape[0], feat_shape[0]))

data_path = os.path.join(os.getcwd(), 'multi_padded')

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

model = load_model('./models/inception_v3_multi.model', custom_objects={'f1': f1})
y_pred = eval_model(model)


'''
d = {'y_true': df.label, 'y_pred': y_pred}
df_results = pd.DataFrame(data=d)
df_results.to_csv('./model_results/inception_v3_drop_4.csv', index=False)

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
'''

import os
import pickle
from keras.models import load_model
from pathlib import Path
from segmentation import Segmentation
from tqdm import tqdm
import numpy as np
import pandas as pd
from shutil import copy
from custom_metrics import f1

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def predict_image(im_path):
    plankton = Segmentation(im_path, target_shape=(75, 75, 3))
    plankton.segment()
    padded = plankton.get_padded()
    feat = plankton.get_features()
    feat = np.array(feat)
    padded = preprocess_input(np.array(padded, dtype=np.float32))
    x_img = padded.reshape(1, padded.shape[0], padded.shape[1], padded.shape[2])
    x_feat = feat.reshape(1, feat.shape[0])
    x_feat = mms.transform(x_feat)
    y_hat = model.predict([x_img, x_feat])
    valid = False
    labels = {}
    results = []
    for i, y in enumerate(y_hat.flatten()):
        if y > 0.5:
            valid = True
            labels[i] = y
    if valid:
        #sort by value. lower prob to higher prob. (given it is above threshold prob)
        labels = sorted(labels.items(), key=lambda x: x[1], reverse=False)
        save_image(im_path, labels)
        label_hat = class_map[labels[0][0]]
        y_hat = [str(x) for x in y_hat.flatten()]
        results.append(im_path)
        results.append(label_hat)
        results.extend(y_hat)
        return results
    return None

def save_image(im_path, labels):
    _, im_name = os.path.split(im_path)
    im_name, ext = im_name.split('.')
    label_hat = class_map[labels[0][0]]
    labels = [class_map[k] for k, v in labels]
    im_name = im_name + '_' + '_'.join(labels) + '.' + ext
    save_path = example_path / 'Classified' / label_hat / im_name
    copy(im_path, save_path)

def check_dirs(wd, dirs):
    for d in dirs:
        exists = os.path.join(wd, d)
        if os.path.isdir(exists) is False:
            os.mkdir(exists)

with open('class_map.p', 'rb') as handle:
    class_map = pickle.load(handle)

with open('normalizer.p', 'rb') as handle:
    mms = pickle.load(handle)

prob_threshold = 0.0

path = Path(os.getcwd())
model_path = path / 'models' / 'inception_v3_multi.model'
example_path = path / 'example_set'
print(class_map)

check_dirs(example_path, ['Classified'])
check_dirs(example_path / 'Classified', list(class_map.values()))
model = load_model(model_path, custom_objects={'f1':f1})

frames = []
columns = ['path', 'predicted_label']
for v in list(class_map.values()):
    columns.append(v)

dirs = next(os.walk(example_path.as_posix()))[1]
dirs.remove('Classified')
extras = ['rawcolor', 'binary']

for d in dirs:
    im_dir = example_path / d / 'images'
    image_dirs = os.listdir(im_dir.as_posix())
    for j in image_dirs:
        image_list = os.listdir(im_dir / j)
        for im in tqdm(image_list):
            if any(x in im for x in extras):
                continue
            im_path = (im_dir / j / im).as_posix()
            results = predict_image(im_path)
            if results:
                frames.append(results)

df = pd.DataFrame(data=frames, columns=columns)
df.to_csv((example_path / 'example_set_results.csv').as_posix(), index=False)

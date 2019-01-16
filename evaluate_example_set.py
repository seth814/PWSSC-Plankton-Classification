import os
import pickle
from keras.models import load_model
from pathlib import Path
from segmentation import Segmentation
from tqdm import tqdm
import numpy as np
import pandas as pd
from shutil import copy

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
    label_hat = class_map[np.argmax(y_hat)]
    save_image(im_path, y_hat)
    return [im_path, label_hat, y_hat]

def save_image(im_path, y_hat):
    _, im_name = os.path.split(im_path)
    ix = np.argmax(y_hat)
    label_hat = class_map[ix]
    prob = y_hat.flatten()[ix]
    if prob < prob_threshold:
        save_path = example_path / 'Classified' / 'Low' / label_hat / im_name
    else:
        save_path = example_path / 'Classified' / 'High' / label_hat / im_name
    copy(im_path, save_path)

def check_dirs(wd, dirs):
    for d in dirs:
        exists = os.path.join(wd, d)
        if os.path.isdir(exists) is False:
            os.mkdir(exists)

with open('class_map.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

with open('normalizer.pickle', 'rb') as handle:
    mms = pickle.load(handle)

prob_threshold = 0.0

path = Path(os.getcwd())
model_path = path / 'models' / 'inception_v3.model'
example_path = path / 'example_set'
print(class_map)

check_dirs(example_path, ['Classified'])
check_dirs(example_path / 'Classified', ['High', 'Low'])
check_dirs(example_path / 'Classified' / 'High', list(class_map.values()))
check_dirs(example_path / 'Classified' / 'Low', list(class_map.values()))
model = load_model(model_path)

frames = []
columns = ['path', 'predicted_label', 'probability']

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
            frames.append(results)

df = pd.DataFrame(data=frames, columns=columns)
df.to_csv((example_path / 'example_set_results.csv').as_posix(), index=False)

from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.layers import GlobalAveragePooling2D, Concatenate
from keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import pickle
from custom_metrics import f1, f1_loss

n_classes = 37
input_shape = (75, 75, 3)
feat_shape = (16,)

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'inception_v3_training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'inception_v3_validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

def get_single_cls_model():

    pretrain_model = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=input_shape)

    input_image = Input(shape=input_shape)
    x = pretrain_model(input_image)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    c1 = Dense(128-feat_shape[0], activation='relu')(x)
    c2 = Input(shape=feat_shape)
    c = Concatenate(axis=-1,)([c1, c2])
    x = Dense(64, activation='relu')(c)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model([input_image, c2], output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-3),
                  metrics=['acc'])
    model.summary()
    return model

def get_multi_cls_model():

    pretrain_model = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=input_shape)

    input_image = Input(shape=input_shape)
    x = pretrain_model(input_image)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    c1 = Dense(128-feat_shape[0], activation='relu')(x)
    c2 = Input(shape=feat_shape)
    c = Concatenate(axis=-1,)([c1, c2])
    x = Dense(64, activation='relu')(c)
    output = Dense(n_classes, activation='sigmoid')(x)

    model = Model([input_image, c2], output)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(1e-3),
                  metrics=['acc', f1, f1_loss])
    model.summary()
    return model

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


with open('class_map.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

df = pd.read_csv('plankton.csv')
df.drop_duplicates(subset='im_name', inplace=True, keep=False)

params = {'n_classes': n_classes,
          'shape': input_shape,
          'feat_shape': feat_shape,
          'batch_size': 64,
          'shuffle': True,
          'multi': False}

# df, class_map = drop_classes(df, labels=[9, 10, 29, 36], class_map=class_map)

frames = []
for c in np.unique(df.label):
    frames.append(df[df.label==c].sample(n=3000, replace=True, random_state=0))
df_sample = pd.concat(frames)

paths = []
labels = []
data_path = os.path.join(os.getcwd(), 'padded')

for im_name, label in zip(df_sample.im_name, df_sample.label):
    im_dir = os.path.join(data_path, class_map[label])
    im_path = os.path.join(im_dir, im_name)
    paths.append(im_path)
    labels.append(to_categorical(y=label, num_classes=n_classes))

paths = np.array(paths)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.05, random_state=0)

checkpoint = ModelCheckpoint('./models/test.model', monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=3, verbose=1, mode='min')

tensorboard = TrainValTensorBoard(write_graph=False)

tg = DataGenerator(paths=X_train, labels=y_train, augment=True, **params)
vg = DataGenerator(paths=X_val, labels=y_val, **params)

model = get_single_cls_model()

model.fit_generator(generator=tg, validation_data=vg,
                    steps_per_epoch=len(tg)/10, validation_steps=len(vg)/10,
                    epochs=100, verbose=1,
                    callbacks=[tensorboard, checkpoint])

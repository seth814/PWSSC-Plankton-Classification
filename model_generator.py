from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.layers import GlobalAveragePooling2D, Concatenate
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import pickle

n_classes = 37
input_shape = (75, 75, 3)
feat_shape = (23,)

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

def get_merged_model():

    pretrain_model = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=input_shape)

    input_image = Input(shape=input_shape)
    x = pretrain_model(input_image)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    c1 = Dense(256, activation='relu')(x)
    c2 = Input(shape=feat_shape)
    c = Concatenate(axis=-1,)([c1, c2])
    x = Dense(128, activation='relu')(c)
    x = Dense(64, activation='relu')(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model([input_image, c2], output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-3),
                  metrics=['acc'])
    model.summary()
    return model

with open('class_map.pickle', 'rb') as handle:
    class_map = pickle.load(handle)

df = pd.read_csv('plankton.csv')
df.drop_duplicates(subset='im_name', inplace=True)

frames = []

for c in list(class_map.keys()):
    frames.append(df[df.label==c].sample(n=3000, replace=True, random_state=0))
df_sample = pd.concat(frames)

paths = []
labels = []
data_path = os.path.join(os.getcwd(), 'data')

for im_name, label in zip(df_sample.im_name, df_sample.label):
    im_dir = os.path.join(data_path, class_map[label])
    im_path = os.path.join(im_dir, im_name)
    paths.append(im_path)
    labels.append(label)
paths = np.array(paths)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.05, random_state=0)

params = {'n_classes': n_classes,
          'shape': input_shape,
          'feat_shape': feat_shape,
          'batch_size': 64,
          'shuffle': True}

checkpoint = ModelCheckpoint('./models/inception_v3_3k_gen_static_val.model', monitor='val_acc', verbose=1, mode='max',
                             save_best_only=True, save_weights_only=False, period=1)

reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=3, verbose=1, mode='min')

tensorboard = TrainValTensorBoard(write_graph=False)

tg = DataGenerator(paths=X_train, labels=y_train, augment=True, **params)
vg = DataGenerator(paths=X_val, labels=y_val, **params)

model = get_merged_model()

model.fit_generator(generator=tg, validation_data=vg,
                    steps_per_epoch=len(tg), validation_steps=len(vg),
                    epochs=1000, verbose=1,
                    callbacks=[tensorboard, checkpoint])

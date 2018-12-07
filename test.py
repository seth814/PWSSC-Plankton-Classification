import pandas as pd
import numpy as np
import cv2
import pickle
import os
from segmentation import Segmentation
from augment import seq
import matplotlib.pyplot as plt

def plot_images(images):
    fig, axes = plt.subplots(nrows=4, ncols=5, sharex=False,
                             sharey=True, figsize=(16,10))
    fig.suptitle('Augmented Plankton (original top left)', size=20)
    i = 0
    for x in range(4):
        for y in range(5):
            axes[x,y].imshow(images[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

data_path = os.path.join(os.getcwd(), 'data')
class_path = os.path.join(data_path, 'Calanus')
im_path = os.path.join(class_path, 'SPC-PWSSC-0-000901-093-2938-1440-404-408.png')

images = []
plankton = Segmentation(im_path)
plankton.segment()
padded = plankton.get_padded()
images.append(padded)
for _ in range(19):
    aug = seq.augment_image(padded)
    images.append(aug)
plot_images(images)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def label_contour_center(image, c):
    # Places a red circle on the centers of contours
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Draw the countour number on the image
    cv2.circle(image,(cx,cy), 3, (0,0,255), -1)
    return (cx,cy)

def overlay_mask(mask, image):
    #make the mask rgb
    bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each.
    #optional depth value set to 0 no need
    img = cv2.addWeighted(bgr_mask, 0.5, image, 0.5, 0)
    return img

def plot_images(images):
    fig, axes = plt.subplots(nrows=6, ncols=6, sharex=False,
                             sharey=True, figsize=(8,6))
    fig.suptitle('Plankton')
    i = 0
    for x in range(6):
        for y in range(6):
            axes[x,y].imshow(images[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.sample(5)
classes = np.unique(df_train.label)
print('n samples: {}'.format(len(classes)))

images = []
for c in classes[:len(classes)-1]:
    path = df_train[df_train.label==c].iloc[0,0]
    im = cv2.imread(path, -1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    images.append(im)

#plot_images(images)
#plt.show()

overlays = []
for im in images:
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3,3), 0)
    edged = auto_canny(blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
    mask_closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    blank = np.zeros(im.shape, dtype=np.uint8)
    _, contours, hierarchy = cv2.findContours(mask_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    top_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    hull = cv2.convexHull(top_contour)
    cv2.drawContours(im, [hull], 0, (0, 255, 0), 2)
    print('Contour Area: {}'.format(cv2.contourArea(hull)))
    center = label_contour_center(im, hull)
    cv2.imshow('Contoured Plankton', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

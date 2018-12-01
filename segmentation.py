import numpy as np
import cv2
from PIL import Image
import mahotas as mt
import os

class Segmentation(object):

    def __init__(self, im_path, target_shape=(299, 299, 3)):
        '''
        im_path: path to image. should include (.png, .jpeg, .tif) extension.
        target_shape: shape to pad image for neural network input. defaults to (299, 299, 3)
        '''
        _, self.im_name = os.path.split(im_path)
        self.im = cv2.imread(im_path, -1)
        self.segmented = None
        self.overlay = None
        self.padded = None
        self.mask = None
        self.features = []
        self.columns = []
        self.target_shape = target_shape

    def segment(self):
        '''
        Cleans and segments the original image.
        '''
        gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh, (5,5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask_closed = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        _, contours, hierarchy = cv2.findContours(mask_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            top_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            self.mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(self.mask, [top_contour], -1, 255, -1)
            self.overlay_mask(self.mask, self.im.copy())
            self.process_contour(top_contour)
        else:
            self.padded = np.zeros(shape=self.target_shape)
            print('Warning: No contour found in image: {}'.format(self.im_name))

    def process_contour(self, top_contour):
        '''
        top_contour: largest contour found in the image
        Uses the largest contour to calculate features from original image.
        An image cropped from the contour is used to calculate hu moments, haralick, and other features.
        '''
        #calculated moments from largest contour
        M = cv2.moments(top_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x_min, x_max, y_min, y_max = self.bounds_from_contour(top_contour, self.im.shape[:2])
        bounds = (x_min, x_max, y_min, y_max)

        #build features based on segmentation
        self.features.append(x_max-x_min)
        self.features.append(y_max-y_min)
        self.features.append(cv2.contourArea(top_contour))
        #self.features.extend([m for m in M.values()])

        #draw segmentation on copy of image
        self.segmented = self.im.copy()
        cv2.drawContours(self.segmented, [top_contour], 0, (0, 255, 0), 1)
        cv2.line(self.segmented, (x_min, y_min), (x_max, y_min), (255,0,255), 1)
        cv2.line(self.segmented, (x_min, y_min), (x_min, y_max), (255,0,255), 1)
        cv2.circle(self.segmented, (cx,cy), 2, (0,0,255), -1)

        #crop to contour -> resize with aspect -> pad to target_shape
        cropped = self.crop_to_contour(np.array(self.im.copy()), bounds)
        cropped_mask = self.crop_to_contour(np.array(self.mask.copy()), bounds)
        #calculate hu moments on cropped mask
        hu_moments = cv2.HuMoments(cv2.moments(cropped_mask)).flatten()
        self.features.extend([hu for hu in hu_moments])
        #compute mahotas features on cropped segmentation
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        textures = mt.features.haralick(gray)
        ht_mean = textures.mean(axis=0)
        self.features.extend([h for h in ht_mean])
        lbp = mt.features.lbp(gray, radius=5, points=8)
        self.features.extend([l for l in lbp])
        pftas = mt.features.pftas(gray)
        self.features.extend([t for t in pftas])
        self.build_column_labels((hu_moments, ht_mean, lbp, pftas))
        #pad cropped to target_shape
        self.padded = self.resize_image(cropped)

    def overlay_mask(self, mask, image):
        '''
        mask: grayscale mask created after contouring.
        image: original image in bgr format.
        calculates the weights sum of two arrays to build silhouette over image.
        '''
        bgr_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        im = cv2.addWeighted(bgr_mask, 0.1, image, 0.9, 0)
        self.overlay = im

    def auto_canny(self, image, sigma=0.33):
        '''
        computes the median of the single channel pixel intensities.
        applies automatic canny edge detection using the computed median.
        '''
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged

    def bounds_from_contour(self, contour, shape):
        '''
        calculates bounds from a contour to create x_length and y_length.
        '''
        x_min, x_max = shape[0], 0
        y_min, y_max = shape[1], 0
        for cord in contour:
            x, y = cord[0][0], cord[0][1]
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
        return x_min, x_max, y_min, y_max

    def crop_to_contour(self, im, bounds):
        '''
        crops im to the bounds of the contour.
        '''
        x_min, x_max, y_min, y_max = bounds
        im = im[y_min:y_max, x_min:x_max]
        return im

    def pad(self, array, reference_shape, offsets):
        """
        array: array to be padded
        reference_shape: tuple of size of ndarray to create
        offsets: list of offsets (number of elements must be equal to the dimension of the array)
        """
        result = np.zeros(reference_shape, dtype=np.uint8)
        # Create a list of slices from offset to offset + shape in each dimension
        insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim]) for dim in range(array.ndim)]
        # Insert the array in the result at the specified offsets
        result[insertHere] = array
        return result

    def resize_image(self, im):
        '''
        im: image to be resized along major axis. maintains aspect ratio.
        If image is a ndarry, it is converted using PIL.
        thumbnail allows for aspect ratio to be maintaied.
        '''
        size = self.target_shape[:2]
        if isinstance(im, np.ndarray):
            im = Image.fromarray(im)
        im.thumbnail(size, Image.LANCZOS)
        im = np.array(im)
        x = (size[0] - im.shape[0]) // 2
        y = (size[1] - im.shape[1]) // 2
        im = self.pad(im, self.target_shape, offsets=[x,y,0])
        return im

    def build_column_labels(self, iterable):
        '''
        Builds columns for computed features.
        This allows user to change feature settings and dynamically build correct feature size.
        '''
        columns = ['x_length', 'y_length', 'area']
        iter_labels = ['hu_', 'haralick_', 'lbp_', 'tas_']
        for i, label in enumerate(iter_labels):
            unpack = [u for u in iterable[i]]
            for z, feat in enumerate(unpack):
                columns.append(label+str(z))
        self.columns = columns

    def raise_segmentation(self):
        raise AttributeError('Plankton has not been segmented. Please call segment method.')

    def get_mask(self):
        if self.mask is None:
            self.raise_segmentation()
        return self.mask

    def get_overlay(self):
        if self.overlay is None:
            self.raise_segmentation()
        overlay_rgb = cv2.cvtColor(self.overlay, cv2.COLOR_BGR2RGB)
        return overlay_rgb

    def get_segmented(self):
        if self.segmented is None:
            self.raise_segmentation()
        segmented_rgb = cv2.cvtColor(self.segmented, cv2.COLOR_BGR2RGB)
        return segmented_rgb

    def get_padded(self):
        if self.padded is None:
            self.raise_segmentation()
        return cv2.cvtColor(self.padded, cv2.COLOR_BGR2RGB)

    def get_features(self):
        if self.features is []:
            self.raise_segmentation()
        return self.features

    def get_columns(self):
        if self.columns is []:
            self.raise_segmentation()
        return self.columns

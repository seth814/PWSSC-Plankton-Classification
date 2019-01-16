from imgaug import augmenters as iaa
import imgaug as ia

#augmenter expects an image in uint8 format

ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        rotate=(-90, 90),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

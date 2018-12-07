from imgaug import augmenters as iaa
import imgaug as ia

ia.seed(1)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.5,
    iaa.GaussianBlur(sigma=(0, 0.2))),
    iaa.Multiply((0.6, 1.4)),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        rotate=(-90, 90),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

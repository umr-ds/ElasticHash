import tempfile
import random
import imageio
import numpy as np
import imgaug.augmenters as iaa
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential


class AugmentationDataGenerator(ImageDataGenerator):
    def __init__(self, seq: iaa.Sequential, *args, jpeg_compression=(60, 85), jpeg_compression_prob=0.5,
                 **kwargs):
        """
        Extends ImageDataGenerator to support imgaug augmentation sequences and JPEG compression within Keras preprocessing pipeline
        :param seq: imgaug.augmenters.Sequential
        :param jpeg_compression: JPEG compression (1-100)
        :param jpeg_compression_prob: Prob for applying JPEG compression
        """
        # assert seq is not None, "Need to specify seq"
        self.seq = seq
        self.compression = iaa.iap.handle_continuous_param(
            jpeg_compression, "compression",
            value_range=(0, 100), tuple_to_uniform=True, list_to_choice=True)
        self.compression_prob = jpeg_compression_prob
        super().__init__(*args, **kwargs)

    def compress_jpg(self, image, seed=None):
        """
        Random JPEG compression adapted for working with float arrays in Keras
        Original code:
        https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmenters/arithmetic.html#compress_jpeg
        :param img:
        :return:
        """

        if image.size == 0:
            return np.copy(image)

        compression = self.compression.draw_sample(random_state=seed)

        has_no_channels = (image.ndim == 2)
        is_single_channel = (image.ndim == 3 and image.shape[-1] == 1)
        if is_single_channel:
            image = image[..., 0]

        assert has_no_channels or is_single_channel or image.shape[-1] == 3, (
                "Expected either a grayscale image of shape (H,W) or (H,W,1) or an "
                "RGB image of shape (H,W,3). Got shape %s." % (image.shape,))

        maximum_quality = 100
        minimum_quality = 1

        # Map from compression to quality used by PIL
        # We have valid compressions from 0 to 100, i.e. 101 possible
        # values
        quality = int(
            np.clip(
                np.round(
                    minimum_quality
                    + (maximum_quality - minimum_quality)
                    * (1.0 - (compression / 101))
                ),
                minimum_quality,
                maximum_quality
            )
        )

        image_pil = array_to_img(image)
        with tempfile.NamedTemporaryFile(mode="wb+", suffix=".jpg") as f:
            image_pil.save(f, quality=quality)
            f.seek(0)
            pilmode = "RGB"
            if has_no_channels or is_single_channel:
                pilmode = "L"
            image = imageio.imread(f, pilmode=pilmode, format="jpeg")
        if is_single_channel:
            image = image[..., np.newaxis]
        return image

    def random_transform(self, x, seed=None):
        """
        Override ImageDataGenerator.random_transform for JPEG compression and imgaug support.
        Augmentation is applied in the following order:
        1. Random JPEG compression
        2. Keras preprocessing augmentation
        3. imgaug Sequential
        :param x:
        :param seed:
        :return:
        """
        if random.random() < self.compression_prob:
            x = self.compress_jpg(x, seed=seed)
        x = super().random_transform(x, seed=seed)
        if self.seq:
            x = self.seq(image=x)
        return x


sometimes = lambda aug: iaa.Sometimes(0.25, aug)

seq = iaa.Sequential([
    sometimes(
        iaa.OneOf([
            # sometimes(iaa.GaussianBlur(sigma=(0, 0.2))),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            # iaa.Crop(percent=(0, 0.1))
            # (iaa.MotionBlur(k=3, angle=[170, 190]))
        ])),
],
)

robust_aug = AugmentationDataGenerator(
    seq=None,
    jpeg_compression=(55, 85),
    jpeg_compression_prob=0.25,
    # rescale=1.0 / 255,
    brightness_range=[0.2, 1.0],
    # May be in Keras:
    rotation_range=7,
    width_shift_range=0.03,
    height_shift_range=0.03,
    shear_range=0.03,
    zoom_range=0.01,
    horizontal_flip=True,
    fill_mode="reflect"  # "nearest"
)


# Augment in Keras
def keras_aug(dim):
    return Sequential(
        [
            preprocessing.RandomRotation(factor=7 / 360.0),
            preprocessing.RandomTranslation(height_factor=0.03, width_factor=0.03),
            preprocessing.RandomFlip(mode="horizontal"),
            preprocessing.RandomContrast(factor=0.1),
            preprocessing.RandomZoom(0.01)
            # preprocessing.RandomCrop(*dim)
        ],
        name="img_augmentation",
    )


if __name__ == "__main__":
    for i in range(30):
        img = load_img("res/Tiefensee.jpg")
        img_arr = img_to_array(img)
        img_arr = robust_aug.random_transform(img_arr)
        img = array_to_img(img_arr)
        img.save("res/Tiefensee_aug_" + str(i) + ".jpg")

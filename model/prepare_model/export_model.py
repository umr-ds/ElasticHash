import os
#gpus = os.environ["NVIDIA_VISIBLE_DEVICES"]
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import Input, Model
from tensorflow.compat.v1.keras.layers import Activation, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, \
    Lambda
from typing import Callable, Tuple
from config import Config as cfg
from util import str2bool
import argparse
import pickle as pkl
import tempfile


# if using tf >=2.0, disable eager execution to use tf.placeholder
# tf.disable_eager_execution()
# tf.disable_v2_behavior()

# Code from https://github.com/sdanaipat/tf-serve-example
class ServingInputReceiver:
    """
    A callable object that returns a
    `tf.estimator.export.ServingInputReceiver`
    object that provides a method to convert
    `image_bytes` input to model.input
    """

    def __init__(
            self, img_size: Tuple[int],
            preprocess_fn: Callable = None,
            input_name: str = "input_1"):
        self.img_size = img_size
        self.preprocess_fn = preprocess_fn
        self.input_name = input_name

    def decode_img_bytes(self, img_b64: str) -> tf.Tensor:
        """
        Decodes a base64 encoded bytes and converts it to a Tensor.
        Args:
            img_bytes (str): base64 encoded bytes of an image file
        Returns:
            img (Tensor): a tensor of shape (width, height, 3)
        """
        img = tf.io.decode_image(
            img_b64,
            channels=3,
            dtype=tf.uint8,
            expand_animations=False
        )
        img = tf.image.resize(img, size=self.img_size)
        img = tf.ensure_shape(img, (*self.img_size, 3))
        img = tf.cast(img, tf.float32)
        return img

    def __call__(self) -> tf.estimator.export.ServingInputReceiver:
        # a placeholder for a batch of base64 string encoded of image bytes
        imgs_b64 = tf.compat.v1.placeholder(
            shape=(None,),
            dtype=tf.string,
            name="image_bytes")

        # apply self.decode_img_bytes() to a batch of image bytes (imgs_b64)
        imgs = tf.map_fn(
            self.decode_img_bytes,
            imgs_b64,
            dtype=tf.float32)

        # apply preprocess_fn if applicable
        if self.preprocess_fn:
            imgs = self.preprocess_fn(imgs)

        return tf.estimator.export.ServingInputReceiver(
            features={self.input_name: imgs},
            receiver_tensors={"image_bytes": imgs_b64}
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Keras Deep Hashing model to tf-serving model')

    parser.add_argument(
        '--keras_weights',
        default="./weights.keras",
        type=str,
        help='Path to keras weights'
    )

    parser.add_argument(
        '--output_dir',
        default="./exported_model",
        type=str,
        help='Path to exported model'
    )

    parser.add_argument(
        '--split_and_permute',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Split coding layer to generate short (64 bit) '
             'and long (256 bit) codes at the same time and permute bit positions according to permutation files'
    )

    parser.add_argument(
        '--permute_64',
        default="64_16_from_256_perm.pkl",
        type=str,
        help='Permutations for 64 bit codes'
    )

    parser.add_argument(
        '--permute_256',
        default="256_16_perm.pkl",
        type=str,
        help='Permutations for 256 bit codes'
    )

    args = parser.parse_args()

    split_and_permute = args.split_and_permute
    output_dir = args.output_dir

    if split_and_permute:
        with open(args.permute_64, "rb") as f:
            p64, _ = pkl.load(f)


            class Permute64Layer(tf.keras.layers.Layer):
                def __init__(self, name="permute64", **kwargs):
                    super(Permute64Layer, self).__init__(name=name, **kwargs)
                    self.permutation = tf.constant(p64)

                def call(self, inputs):
                    return tf.gather(inputs, self.permutation, axis=1)

                def get_config(self):
                    config = super(Permute64Layer, self).get_config()
                    return config

        with open(args.permute_256, "rb") as f:
            p256, _ = pkl.load(f)


            class Permute256Layer(tf.keras.layers.Layer):
                def __init__(self, name="permute64", **kwargs):
                    super(Permute256Layer, self).__init__(name=name, **kwargs)
                    self.permutation = tf.constant(p256)

                def call(self, inputs):
                    return tf.gather(inputs, self.permutation, axis=1)

                def get_config(self):
                    config = super(Permute256Layer, self).get_config()
                    return config

    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:' + str(gpu) for gpu in range(len(gpus))])

    dim = cfg.IMG_HEIGHT, cfg.IMG_WIDTH  # h, w #(260, 260) #(380, 380)
    input_shape = dim + (3,)

    with strategy.scope():
        input_tensor = Input(shape=input_shape)
        backbone = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights=None,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=None,
        )
        x = backbone.output
        x = GlobalAveragePooling2D()(x)  # pool_size=(5, 5), strides=(5, 5))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2, name="top_dropout")(x)
        x = Dense(256, name='embedding', activation=None,
                  kernel_initializer=tf.keras.initializers.random_normal(stddev=1e-2))(x)
        # t = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="l2_code")(x) # L2 normalize embeddings

        if split_and_permute:
            # TF 1.x only:
            # code64_perm = Lambda(lambda t: tf.keras.backend.gather(x, p64), name="code64")(x)
            # code256_perm = Lambda(lambda t: tf.keras.backend.gather(x, p256), name="code256")(x)
            code64_perm = Permute64Layer(name="code64")(x)
            code256_perm = Permute256Layer(name="code256")(x)
            model = Model(inputs=backbone.input, outputs=[code64_perm, code256_perm])
        else:
            code = Activation(activation=tf.keras.activations.tanh, name="code")(x)
            model = Model(inputs=backbone.input, outputs=[code])

        model.load_weights(args.keras_weights, by_name=True)
        model.compile(loss="categorical_crossentropy")

        MODEL_DIR = tempfile.gettempdir()
        version = 1
        export_path = "exported_model/"
        print('export_path = {}\n'.format(export_path))

        serving_input_receiver = ServingInputReceiver(
            img_size=(300, 300),
            preprocess_fn=tf.keras.applications.efficientnet.preprocess_input,
            input_name="input_1")

        estimator_save_dir = os.path.join(output_dir, "estimator")

        if split_and_permute:
            custom_objects = {"Permute64Layer": Permute64Layer, "Permute256Layer": Permute256Layer}
        else:
            custom_objects = None
        estimator = tf.keras.estimator.model_to_estimator(
            keras_model=model,
            model_dir=estimator_save_dir,
            custom_objects=custom_objects
        )

        export_model_dir = os.path.join(output_dir, "model")
        estimator.export_saved_model(
            export_dir_base=export_model_dir,
            serving_input_receiver_fn=serving_input_receiver)

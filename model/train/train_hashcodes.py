import os
import numpy as np
from callbacks import FixedModelCheckPoint
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from augment import robust_aug
from generators import TrainClassificationDataGenerator, \
    ValClassificationDataGenerator
from util import create_dirs
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras import Model

# Tips for finetuning Efficientnet:
# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_efficientnet_fine_tuning.ipynb#scrollTo=dgmY5Od06tDo

# Training settings
n_epochs = 30  # number of epochs
gpus = [0, 1, 2, 3, 4, 5]
batch_size = 64  # 80: top layers
global_batch_size = batch_size * len(gpus)
dim = (300, 300)  # h, w #(260, 260) #(380, 380)
input_shape = dim + (3,)
save_freq = 2000  # Number of batches for checkpoint

train_datagen = TrainClassificationDataGenerator(aug=robust_aug, preprocess=preprocess_input,
                                                 batch_size=global_batch_size, dim=dim)
val_datagen = ValClassificationDataGenerator(batch_size=10, preprocess=preprocess_input, dim=dim,
                                             val_set='./val_np_array.pkl')

num_imgs_per_class = train_datagen.imgs_per_class
num_classes = train_datagen.num_classes
epoch_size = np.floor(num_imgs_per_class * num_classes / global_batch_size)


def unfreeze_model(model, layer_num, optimizer, no_batch_norm=True):
    print("Unfreezing the following layers: ")
    print(", ".join([layer.name for layer in model.layers]))  # [-layer_num]]))
    for layer in model.layers:  # [-layer_num]:  # -20:
        if not no_batch_norm or not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy']
    )


# Create log and checkpoint dirs
base_dir = './'
log_dir, chkpt_dir = create_dirs(base_dir)

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

# Multi GPU training
strategy = tf.distribute.MirroredStrategy(devices=['/device:GPU:' + str(gpu) for gpu in gpus]
                                          # ,      cross_device_ops=tf.distribute.ReductionToOneDevice()
                                          )
with strategy.scope():
    input_tensor = Input(shape=input_shape)
    backbone = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=None,
        # Load best weights from https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b3.tar.gz
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=None,  # 'avg',
        # drop_connect_rate = 0.4
    )
    backbone.load_weights('./efficientnetb3_notop_noisy_student.h5')
    # Freeze the pretrained weights
    backbone.trainable = False
    x = backbone.output
    x = GlobalAveragePooling2D()(x)  # pool_size=(5, 5), strides=(5, 5))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2, name="top_dropout")(x)

    # Coding layers
    # c1 = Dense(64, name="short_code")(x)
    c = Dense(256, name="code", kernel_initializer='xavier')(x)
    output = Dense(num_classes, name='pred')(c)
    # output_c1 = Dense(num_classes, name='pred_short', kernel_initializer='xavier')(c1)  # activation='softmax',
    model = Model(inputs=backbone.input, outputs=[output])

    # Write checkpoints
    cb_checkpoint = FixedModelCheckPoint(
        filepath=os.path.join(base_dir, chkpt_dir, "weights.e{epoch:003d}b{batch:0000000003d}l{loss:.2f}.keras"),
        monitor='loss',
        verbose=0,
        save_best_only=False, save_weights_only=False,
        mode='auto', save_freq=save_freq)

    # Finetune model for 1-2 epochs
    # model.compile(
    #    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss=loss, metrics=["accuracy"]
    # )
    #
    # print(model.summary())
    #
    #
    #
    # model.fit(train_datagen,
    #           validation_data=val_datagen,
    #           workers=16, max_queue_size=16, use_multiprocessing=True,
    #           epochs=2,
    #           steps_per_epoch=epoch_size,
    #           callbacks=[
    #               cb_checkpoint,
    #               tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    #           ])

    # model = tf.keras.models.load_model("./checkpoints_2020_10_25_170837/weights.e002b13964l3.00.keras")
    #
    # unfreeze_model(model, len(model.layers),
    #                tf.keras.optimizers.Adam(learning_rate=1e-4)
    #                # RMSprop(learning_rate=2e-5, momentum=0.9, epsilon=1e-07, centered=False, decay=0.97)
    #                )
    #
    # model.fit(train_datagen,
    #           validation_data=val_datagen,
    #           workers=16, max_queue_size=16, use_multiprocessing=True,
    #           epochs=40,
    #           steps_per_epoch=epoch_size,
    #           callbacks=[
    #               cb_checkpoint,
    #               tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    #           ])

    model = tf.keras.models.load_model("./checkpoints_2020_10_26_100635/weights.e016b13460l0.09.keras")

    unfreeze_model(model, len(model.layers),
                   tf.keras.optimizers.Adam(learning_rate=1e-4), no_batch_norm=False
                   # RMSprop(learning_rate=2e-5, momentum=0.9, epsilon=1e-07, centered=False, decay=0.97)
                   )

    model.fit(train_datagen,
              validation_data=val_datagen,
              workers=16, max_queue_size=16, use_multiprocessing=True,
              epochs=40,
              steps_per_epoch=epoch_size,
              callbacks=[
                  cb_checkpoint,
                  tf.keras.callbacks.TensorBoard(log_dir=log_dir)
              ])

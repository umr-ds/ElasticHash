import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from augment import robust_aug, AugmentationDataGenerator
from util import resize_image, one_hot


class TrainClassificationDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 batch_size,
                 preprocess,
                 aug: AugmentationDataGenerator = robust_aug,
                 dim=(300, 300),  # (h, w)
                 label_map="./labels.txt",
                 train_list="./train.pkl",
                 data_dir="/data/",
                 epoch_size=None, one_hot=True,
                 num_samples_per_class=2):
        """

        :param batch_size:
        :param preprocess: architecture specific preprocessing function
        :param aug:
        :param dim: (height, width) for network input
        :param label_map: path to label map
        :param train_list: pickled dict containing image pathes per class
        :param data_dir:
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.dim = dim
        self.aug = aug
        self.preprocess = preprocess
        self.n_channels = 3
        self.epoch_size = epoch_size
        self.num_samples_per_class = num_samples_per_class

        # Load training data and labels
        with open(label_map, "r") as f:
            self.cls_map = {}
            for l in f.readlines():
                cls, cls_str = l.strip().split(" ", 1)
                self.cls_map[int(cls)] = cls_str
        with open(train_list, "rb") as f:
            self.train = pkl.load(f)

        self.counts = [len(imgs) for imgs in self.train.values()]
        self.total_images = sum(self.counts)
        self.num_classes = len(self.train)
        self.imgs_per_class = 1000
        self.size = self.imgs_per_class * self.num_classes
        self.one_hot = one_hot

    def __len__(self):
        if (self.epoch_size):
            return self.epoch_size
        else:
            return int(np.floor(self.size / float(self.batch_size)))

    def __getitem__(self, idx):
        # Init batch
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        if self.one_hot:
            y_true = np.empty((self.batch_size, self.num_classes), dtype=np.float32)
        else:
            y_true = np.empty(self.batch_size, dtype=np.int)

        # Sample classes
        num_unique_classes = int(np.ceil(self.batch_size / self.num_samples_per_class))
        classes = np.random.choice(range(0, self.num_classes), replace=False, size=num_unique_classes)
        classes = np.tile(classes, self.num_samples_per_class)
        classes = classes[:self.batch_size]
        # Sample, augment, and preprocess images
        for i, cls in enumerate(classes):
            img_processed = 0
            while not img_processed:
                img_path = self.train[cls][np.random.randint(0, self.counts[cls])]
                try:
                    img = load_img(os.path.join(self.data_dir, img_path))
                except:
                    tf.compat.v1.logging.warn("Could not load %s" % (img_path,))
                    continue
                img_arr = img_to_array(img)
                img_arr = resize_image(img_arr, *self.dim, resize_mode='random_crop')
                # if self.aug:
                #     img_arr = self.aug.random_transform(img_arr)
                img_arr = self.preprocess(img_arr)
                X[i, :] = img_arr
                if self.one_hot:
                    y_true[i, :] = one_hot(cls, self.num_classes)
                else:
                    y_true[i] = cls
                img_processed = 1

        return X, y_true


def get_train_dataset(datagen_train, workers=6, max_queue_size=6):
    def ds_train():
        multi_enqueuer = tf.keras.utils.OrderedEnqueuer(datagen_train, use_multiprocessing=False)
        multi_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
        while True:
            batch_xs, batch_ys = next(multi_enqueuer.get())
            yield batch_xs, batch_ys

    train_tf_ds = tf.data.Dataset.from_generator(ds_train, output_types=(tf.float32, tf.float32), output_shapes=(
        tf.TensorShape([datagen_train.batch_size, *datagen_train.dim, 3]),
        tf.TensorShape([datagen_train.batch_size, 1])
    ))

    return train_tf_ds


class ValClassificationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 batch_size,
                 preprocess,
                 num_classes,
                 dim=(300, 300),  # (h, w)
                 val_list="./val.pkl",
                 data_dir="/data/",
                 val_set="./val_np_array.pkl", load_processed_dataset=True, one_hot=True):
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = 3
        self.one_hot = one_hot
        self.num_classes = num_classes

        if val_set and os.path.isfile(val_set) and load_processed_dataset:
            with open(val_set, "rb") as f:
                self.X, self.y_true = pkl.load(f)
            tf.compat.v1.logging.info("Loaded dataset from %s" % val_set)
        elif val_list and os.path.isfile(val_list):
            with open(val_list, "rb") as f:
                val_list = pkl.load(f)

            img_pathes = []
            classes = []

            # Read image pathes
            for cls in val_list:
                for path in val_list[cls]:
                    img_pathes += [path]
                    classes += [cls]

            num_images = len(classes)
            tf.compat.v1.logging.info("Preparing validation set (%d images)..." % (num_images,))

            # X = np.empty((num_images, *dim, 3), dtype=np.float32)
            # y_true = np.empty((num_images,), dtype=np.float32)
            X = []
            y_true = []

            pb = tf.keras.utils.Progbar(
                num_images, width=30, verbose=1, interval=0.05, stateful_metrics=None,
                unit_name='step'
            )

            for i in range(len(img_pathes)):
                try:
                    img_path = img_pathes[i]
                    img = load_img(os.path.join(data_dir, img_path))
                except:
                    tf.compat.v1.logging.warn("Could not load %s" % (img_path,))
                    continue
                img_arr = img_to_array(img)
                img_arr = resize_image(img_arr, *self.dim, resize_mode='crop')
                img_arr = preprocess(img_arr)
                X.append(img_arr)
                y_true.append(classes[i])
                pb.update(i)

            tf.compat.v1.logging.info("%d images loaded" % (len(y_true),))
            if self.one_hot:
                self.y_true = np.array(y_true, dtype=np.float32)
            else:
                self.y_true = np.array(y_true, dtype=np.int)
            self.X = np.array(X, dtype=np.float32)
            tf.compat.v1.logging.info("Saving to %s" % (val_set,))
            with open(val_set, "wb") as f:
                pkl.dump((self.X, self.y_true), f, protocol=4)

        else:
            raise ValueError("Specify either valid val_list or val_set")

    def __len__(self):
        return int(np.floor(len(self.y_true) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Init batch
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        if self.one_hot:
            y_true = np.empty((self.batch_size, self.num_classes), dtype=np.float32)
        else:
            y_true = np.empty(self.batch_size, dtype=np.int)

        # Fill
        for i, id in enumerate(range(idx, min(idx + self.batch_size, len(self.y_true)))):
            X[i, :] = self.X[id, :]
            if self.one_hot:
                y_true[i, :] = one_hot(self.y_true[id], self.num_classes)
            else:
                y_true[i] = self.y_true[id]
        return X, y_true


if __name__ == '__main__':
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # d = TrainClassificationDataGenerator(batch_size=2, aug=robust_aug, preprocess=preprocess_input)
    d = ValClassificationDataGenerator(batch_size=10,
                                       preprocess=preprocess_input,
                                       val_list='./val.pkl',
                                       load_processed_dataset=False, num_classes=5390)

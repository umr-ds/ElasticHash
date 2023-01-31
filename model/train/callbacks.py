import random
import six
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.distribute import distributed_file_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
import pickle as pkl
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.layers import InputLayer
from tensorboard.plugins import projector
from util import save_labels_tsv, euclidean_distance_np, images_to_sprite, log_histogram


# Fixes missing batch variable in formatting parameters, see:
# https://github.com/tensorflow/tensorflow/issues/38668
class FixedModelCheckPoint(ModelCheckpoint):

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        # pylint: disable=protected-access
        if self.save_freq == 'epoch':
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _save_model(self, epoch, batch, logs):
        """Saves the model.
        Arguments:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
              is set to `epoch`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
                """
        logs = logs or {}
        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            logs = tf_utils.to_numpy_or_python_type(logs)
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, batch, logs)
            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)
                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""
        # pylint: disable=protected-access
        try:
            # `filepath` may contain placeholders such as `{epoch:02d}` and
            # `{batch:02d}`. A mismatch between logged metrics and the path's
            # placeholders can cause formatting to fail.
            if not batch:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch + 1,
                    batch=batch + 1,
                    **logs)
        except KeyError as e:
            raise KeyError('Failed to format this callback filepath: "{}". '
                           'Reason: {}'.format(self.filepath, e))
        self._write_filepath = distributed_file_utils.write_filepath(
            file_path, self.model.distribute_strategy)
        return self._write_filepath


class SGDRScheduler(Callback):
    """
    Apply a one cycle cosine learning rate decay schedule from "learning_rate" to "alpha" with "decay_steps" steps,
    where "global_step" is the current step
    # Arguments
        learning_rate: maximal learning rate (learning rate to start schedule from)
        decay_steps: total number of steps of the schedule
        global_step: number of the current step of the schedule
        alpha: minimal learning rate (learning rate to end schedule with)
    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 global_step=0,
                 alpha=0.0
                 ):
        super(SGDRScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.global_step = global_step
        self.alpha = alpha

    def clr(self):
        cosine_decay = 0.5 * (1 + np.cos(np.pi * self.global_step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        decayed_learning_rate = self.learning_rate * decayed
        return decayed_learning_rate

    def on_train_end(self, logs=None):
        # Initialize the learning rate to the minimum value at the start of training
        K.set_value(self.model.optimizer.lr, self.learning_rate)

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        K.set_value(self.model.optimizer.lr, self.clr())


class ExtendedTensorboardStats(TensorBoard):
    def __init__(self,
                 val_list="./val.pkl",
                 data_dir="/data/",
                 # csv_file='/images/Dataset_Reidentification/val_topview.csv',
                 # image_path='/images/Dataset_Reidentification/val',
                 dim=(300, 300),
                 preprocess=False,
                 log_dir='./logs',
                 batch_size=32,
                 update_freq='epoch',
                 embedding_layer_name='code',
                 num_vis_images=40,
                 weight_layers=None,
                 # num_db_classes=400,
                 num_query_imgs=200,
                 num_images_per_class=3,
                 one_hot=True,
                 label_map="./labels.txt",
                 code_len=256,
                 **kwargs):
        super(ExtendedTensorboardStats, self).__init__(log_dir=log_dir,
                                                       # batch_size=batch_size,
                                                       update_freq=update_freq,
                                                       # Important: deactivate profiling, see https://github.com/tensorflow/tensorboard/issues/2084
                                                       profile_batch=0,
                                                       **kwargs)

        # assert (num_query_classes <= num_db_classes, "Database needs to be larger then query set")

        self.weight_layers = weight_layers
        self.embedding_layer_name = embedding_layer_name
        self.dim = dim
        self.image_path = data_dir
        self.preprocess = preprocess
        self.batch_size = batch_size
        self.num_images_per_class = num_images_per_class
        self.num_vis_images = num_vis_images
        self.code_len = code_len

        seed = 1
        np.random.seed(seed=seed)

        # Read data

        with open(label_map, "r") as f:
            self.cls_map = {}
            for l in f.readlines():
                cls, cls_str = l.strip().split(" ", 1)
                self.cls_map[int(cls)] = cls_str

        if val_list and os.path.isfile(val_list):
            with open(val_list, "rb") as f:
                val_list = pkl.load(f)

            img_pathes = []
            classes = []

            # Read image pathes
            for cls in val_list:
                for path in val_list[cls]:
                    img_pathes += [path]
                    # if (one_hot):
                    #    cls = np.argmax(cls)
                    classes += [cls]

            # Shuffle
            zipped = list(zip(classes, img_pathes))
            random.shuffle(zipped)
            classes, img_pathes = zip(*zipped)

        # Prepare database and queries
        self.labels_q = np.array(classes[:num_query_imgs])
        self.labels_db = np.array(classes[num_query_imgs:])
        self.size_db = len(self.labels_db)
        self.size_q = len(self.labels_q)

        self.img_paths_q, self.img_paths_db = img_pathes[:num_query_imgs], img_pathes[num_query_imgs:]
        imgs = [self._load_image(i, preprocess=self.preprocess) for i in img_pathes]
        self.imgs_q, self.imgs_db = imgs[:num_query_imgs], imgs[num_query_imgs:]

        # Prepare triplets visualization
        self.vis_ids = np.random.choice(range(self.size_q), size=num_vis_images, replace=False)
        self.vis_labels = [self.labels_q[i] for i in self.vis_ids]
        self.vis_pathes = [self.img_paths_q[i] for i in self.vis_ids]

        # Prepare embeddings visualization
        self.num_query_imgs_for_emb = min(30, num_query_imgs)
        self.num_db_imgs_for_emb = self.num_query_imgs_for_emb * num_images_per_class
        self.labels_emb = np.append(self.labels_q[:self.num_query_imgs_for_emb],
                                    self.labels_db[:self.num_db_imgs_for_emb], axis=0)
        self.img_paths_emb = np.append(self.img_paths_q[:self.num_query_imgs_for_emb],
                                       self.img_paths_db[:self.num_db_imgs_for_emb], axis=0)

        # Write sprites image
        num_img_sprite = len(self.img_paths_emb)
        self.sprite_size = 299  # int(np.floor(np.sqrt((8000.0**2)/num_img_sprite))) # 8192 max size for tensorboard sprite image
        # if (self.sprite_size > self.dim[0]):
        #    self.sprite_size = self.dim

        imgs_sprite = [
            self._load_image(i, preprocess=False, resize=(self.sprite_size, self.sprite_size), swap_channels=True) for i
            in self.img_paths_emb]
        imgs_sprite = np.array(imgs_sprite)
        sprite = images_to_sprite(imgs_sprite)
        self.sprite_img_path = os.path.join(self.log_dir, 'sprite.jpg')
        cv2.imwrite(self.sprite_img_path, sprite)

        # Write labels file
        self.labels_filename = 'metadata.tsv'
        self.ckpt_path = os.path.join(self.log_dir, self.embedding_layer_name + '.ckpt')
        save_labels_tsv(self.labels_emb, self.img_paths_emb, self.labels_filename, self.log_dir)

    def _load_image(self, basename, preprocess=False, resize=False, swap_channels=False, target_size=None):
        if not target_size:
            target_size = self.dim
        image = load_img(os.path.join(self.image_path, basename), target_size=target_size,
                         interpolation='bilinear')
        if resize:
            image = image.resize(resize)
        image = img_to_array(image)
        if swap_channels:
            image = image[..., ::-1]
        if preprocess:
            image = preprocess(image)
        return image

    def _predict_batch(self, imgs, start, size):
        bs = self.batch_size
        X = np.empty((self.batch_size, *self.dim, 3))
        if (start + bs < size):
            X[0:bs, ] = imgs[start:start + bs]
            return self.embedding_model.predict_on_batch(X)
        else:  # fill batch
            X[0:bs, ] = imgs[0:bs]
            X[0:size - start, ] = imgs[start:size]
            return self.embedding_model.predict_on_batch(X)[0:size - start]

    def _predict(self, imgs, size, ds="", output=None):
        if not output:
            output = self.embedding_layer_name
        self.embedding_model = Model(inputs=self.model.input,
                                     outputs=self.model.get_layer(output).output)
        vecs = []
        for i in range(0, size, self.batch_size):
            print('Predicting %s (%d / %d)' % (ds, i, size), end='\r')
            vecs += self._predict_batch(imgs, i, size).tolist()
        vecs = np.array(vecs)
        return vecs

    def _mean_average_precision(self, dists):

        aps = []
        for i, l in enumerate(self.labels_q):
            gt = np.zeros(self.size_db)
            gt[np.argwhere(self.labels_db == l)] = 1.
            pred = dists[:, i]
            ap = average_precision_score(gt, pred)
            aps += [ap]
        self.aps = aps
        return np.mean(aps), aps

    def _accuracy(self, dists):
        tp = 0.
        for i, l in enumerate(self.labels_q):
            dists_i = dists[:, i]
            pos_dists = np.copy(dists_i)
            pos_dists[np.argwhere(self.labels_db != l)] = np.inf
            nearest_pos_id = np.argmin(pos_dists)
            d_p = pos_dists[nearest_pos_id]
            neg_dists = np.copy(dists_i)
            neg_dists[np.argwhere(self.labels_db == l)] = np.inf
            nearest_neg_id = np.argmin(neg_dists)
            d_n = neg_dists[nearest_neg_id]
            if (d_p <= d_n):
                tp += 1
        return tp / len(self.labels_q)

    def _visualize_triplet(self, img_a, img_p, img_n, d_p, d_n, neg_text="", cls_text=""):
        font = ImageFont.truetype("arial.ttf", 25)
        w = 400
        h = 300
        size = w, h
        a = Image.fromarray(img_a.astype('uint8')).resize(size)
        p = Image.fromarray(img_p.astype('uint8')).resize(size)
        n = Image.fromarray(img_n.astype('uint8')).resize(size)
        image = Image.new('RGB', (1200, 420))
        image.paste(a, (0, 0))
        image.paste(p, (w, 0))
        image.paste(n, (2 * w, 0))
        color_a = (255, 255, 255, 255)
        color_p = (0, 255, 0, 255)
        color_n = (255, 0, 0, 255)
        offset = 5
        d = ImageDraw.Draw(image)
        d.text((offset * 4, h + offset), "anchor", fill=color_a, font=font)
        d.text((offset * 4, 365), neg_text + " (class of nearest negative)\nAP: " + cls_text, fill=(255, 255, 0, 255),
               font=font)
        d.text((offset * 4 + w, h + offset),
               "nearest positive\nd:   " + str(d_p), fill=color_p,
               font=font)
        d.text((offset * 4 + 2 * w, h + offset),
               "nearest negative\nd:   " + str(d_n), fill=color_n,
               font=font)
        if d_p < d_n:
            color_box_short = color_p
        else:
            color_box_short = color_n
        d.rectangle([offset * 3, h + offset + 30, offset * 3 + 60, h + offset + 60], fill=color_box_short, outline=None)
        return np.array(image)

    def _register_embedding(self, embedding_tensor_name, meta_data_path, log_dir, ckpt_path):
        config = projector.ProjectorConfig()
        config.model_checkpoint_path = ckpt_path
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_tensor_name
        embedding.metadata_path = meta_data_path
        embedding.sprite.image_path = self.sprite_img_path
        embedding.sprite.single_image_dim.extend((self.sprite_size, self.sprite_size))
        projector.visualize_embeddings(log_dir, config)

    def on_epoch_end(self, epoch, logs=None):

        super(ExtendedTensorboardStats, self).on_epoch_end(epoch, logs)

        print("\nEvaluation started", end='\r')

        q_vecs = self._predict(self.imgs_q, self.size_q, "queries")
        db_vecs = self._predict(self.imgs_db, self.size_db, "database")

        # Hamming
        q_vecs = np.sign(q_vecs)
        db_vecs = np.sign(db_vecs)

        self.dists = (self.code_len - (0.5 * (self.code_len + np.dot(db_vecs,
                                                                     q_vecs.T)))) / self.code_len  # euclidean_distance_np(db_vecs, q_vecs) #-np.dot(db_vecs, q_vecs.T)  # euclidean_distance_np(db_vecs, q_vecs)

        # MAP
        mean_ap, _ = self._mean_average_precision(1.0 - self.dists)
        map_string = "Validation mAP = {:.3f} (for {:d} queries on {:d} db images)".format(mean_ap, self.size_q,
                                                                                           self.size_db)
        with self._val_writer.as_default():
            tf.summary.scalar("map (codes)", mean_ap, description=map_string, step=epoch)
        print(map_string)

        # Accuracy
        acc = self._accuracy(self.dists)
        acc_string = "Top-1 validation accuracy = {:.3f} (for {:d} queries on {:d} db images)".format(acc, self.size_q,
                                                                                                      self.size_db)
        with self._val_writer.as_default():
            tf.summary.scalar("accuracy (codes)", acc, description=acc_string, step=epoch)

        # Embeddings
        x = np.append(q_vecs[:self.num_query_imgs_for_emb], db_vecs[:self.num_db_imgs_for_emb], axis=0)
        tensor_embeddings = tf.Variable(x, name=self.embedding_layer_name)
        saver = tf.compat.v1.train.Saver([tensor_embeddings])
        saver.save(sess=None, global_step=0, save_path=self.ckpt_path)
        ckpt = tf.train.Checkpoint(embeddings=tensor_embeddings)
        ckpt.write(self.ckpt_path)
        reader = tf.train.load_checkpoint(self.ckpt_path)
        varmap = reader.get_variable_to_shape_map()
        key_to_use = ""
        for key in varmap:
            # Workaround, see: https://github.com/tensorflow/tensorboard/issues/2471
            if "embeddings" in key:
                key_to_use = key
        self._register_embedding(key_to_use, self.labels_filename, self.log_dir, self.ckpt_path)

        # Image summaries for triplet visualization
        for i, l, p in zip(self.vis_ids, self.vis_labels, self.vis_pathes):
            img_a = self._load_image(self.img_paths_q[i], preprocess=False)

            dists_a = self.dists[:, i] * self.code_len
            neg_dists = np.copy(dists_a)
            neg_dists[np.argwhere(self.labels_db == l)] = np.inf
            nearest_neg_id = np.argmin(neg_dists)
            img_n = self._load_image(self.img_paths_db[nearest_neg_id], preprocess=False)
            d_n = neg_dists[nearest_neg_id]

            pos_dists = np.copy(dists_a)
            pos_dists[np.argwhere(self.labels_db != l)] = np.inf
            nearest_pos_id = np.argmin(pos_dists)
            img_p = self._load_image(self.img_paths_db[nearest_pos_id], preprocess=False)
            d_p = pos_dists[nearest_pos_id]

            img = self._visualize_triplet(img_a, img_p, img_n, d_p, d_n,
                                          neg_text=self.cls_map[self.labels_db[nearest_neg_id]],
                                          cls_text=str(self.aps[i]))

            img = img[np.newaxis, :]  # add batch dim
            with self._val_writer.as_default():
                tf.summary.image(self.cls_map[l], img, step=epoch)

        # Weights histograms
        def write_hist_summary(weight_name, weight, epoch):
            log_histogram(self._val_writer, weight_name, weight, epoch, bins=1000)

        if self.weight_layers is None:
            layers = [self.model.get_layer(layer.name)
                      for layer in self.model.layers
                      if not isinstance(layer, InputLayer)
                      ]
        else:
            layers = [self.model.get_layer(layer)
                      for layer in self.weight_layers
                      ]
        for layer in layers:
            weights = layer.get_weights()
            if weights:
                weight_names = layer.weights
                for v, weight in zip(weight_names, weights):
                    weight_name = v.name.replace(":", "_")
                    write_hist_summary(weight_name, weight, epoch)

        self._val_writer.flush()

        # Learning rate summary
        with self._train_writer.as_default():
            tf.summary.scalar("learning rate", self.model.optimizer.lr, description="learning rate", step=epoch)

        self._train_writer.flush()

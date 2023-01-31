import tensorflow as tf
import numpy as np
import pickle as pkl


class TrainDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 batch_size,
                 label_map="./labels.txt",
                 train_list="./train.pkl",
                 data_dir="/data/"):
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Load training data and labels
        with open(label_map, "r") as f:
            self.cls_map = {}
            for l in f.readlines():
                cls, cls_str = l.strip().split(" ", 1)
                self.cls_map[int(cls)] = cls_str
        with open(train_list, "rb") as f:
            self.train = pkl.load(f)

        counts = [len(imgs) for imgs in self.train.values()]
        self.total_images = sum(counts)
        self.num_classes = len(self.train)
        self.imgs_per_class = 1000
        self.size = self.imgs_per_class * self.num_classes

    def __len__(self):
        return int(np.ceil(self.size / float(self.batch_size)))

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    d = TrainDataGenerator(batch_size=1)  # DataSetBuilder()

# with open(label_counts, "r") as f:
#     self.cls_count = {}
#     for l in f.readlines():
#         cls, count, _ = l.strip().split(" ", 2)
#         self.cls_count[int(cls)] = int(count)

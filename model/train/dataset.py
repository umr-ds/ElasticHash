import os
import random
import pickle as pkl
from logging import info as log


def insert_img(d, cls, img):
    if cls in d:
        d[cls] += [img]
    else:
        d[cls] = [img]


class ImageNetPlacesBuilder():
    def __init__(self,
                 val_size=5,
                 label_map="./labels.txt",
                 label_counts="./counts.txt",
                 val_list="./val.pkl",
                 train_list="./train.pkl"):
        self.data_dir = "/data/"
        self.val_size = val_size

        # ImageNet
        self.imagenet_dir = "ImageNet/unpacked_more_than_1000/"
        self.imagenet_label_file = "ImageNet/leaf_nodes_more_than_1000_with_names.txt"

        # Places
        self.places_label_files = {"365": "Places/categories_places365.txt",
                                   "69": "Places/categories_extra69.txt"}

        self.places_image_lists = {"train": "Places/data/places365_train_challenge.txt",
                                   "extra": "Places/imglist_extra_train_and_test.txt",
                                   "val": "Places/data/places365_val.txt"}

        self.places_dirs = {"train": "Places/data_large/",
                            "extra": "Places/data_large_extra/",
                            "val": "Places/val_large/"}

        self.cls_imgs, self.cls_map, self.cls_count = self._read_data()
        self._split_data()
        with open(label_map, "w") as f:
            for cls, cls_str in self.cls_map.items():
                f.write(str(cls) + " " + cls_str + "\n")
        with open(label_counts, "w") as f:
            for cls, count in self.cls_count.items():
                f.write(str(cls) + " " + str(count) + " " + self.cls_map[cls] + "\n")
        with open(train_list, "wb") as f:
            pkl.dump(self.train, f)
        with open(val_list, "wb") as f:
            pkl.dump(self.val, f)

    def _split_data(self):
        assert self.cls_imgs is not None and self.cls_map is not None, "Read data first"
        self.train = {}
        self.val = {}
        for cls in self.cls_map:
            random.shuffle(self.cls_imgs[cls])
            val, train = self.cls_imgs[cls][:self.val_size], self.cls_imgs[cls][self.val_size:]
            self.train[cls] = train
            self.val[cls] = val

    def _read_data(self):
        log("Processing data...")
        cls_imgs = {}
        cls_map = {}
        cls_count = {}
        current_cls = 0

        # Prepare ImageNet
        with open(os.path.join(self.data_dir, self.imagenet_label_file), "r") as f:
            imagenet_label_map = {node: class_str for node, class_str in
                                  [l.strip().split(" ", 1) for l in f.readlines()]}
        for cls in imagenet_label_map:
            cls_map[current_cls] = imagenet_label_map[cls]  # Add label
            imgs = [os.path.join(self.imagenet_dir, cls, img) for img in
                    os.listdir(os.path.join(self.data_dir, self.imagenet_dir, cls)) if img.endswith(".JPEG")]
            cls_imgs[current_cls] = imgs  # Add images
            cls_count[current_cls] = len(imgs)
            current_cls += 1

        # Prepare Places 365
        with open(os.path.join(self.data_dir, self.places_label_files["365"])) as f:
            places_365_label_map = {int(cls): cls_str for cls_str, cls in
                                    [l.strip().split(" ", 1) for l in f.readlines()]}
            num_places_365_cls = len(places_365_label_map)
            places_365_gcls_map = {cls: cls + current_cls for cls in range(num_places_365_cls)}
            current_cls += num_places_365_cls
            for cls, gcls in places_365_gcls_map.items():
                cls_map[gcls] = places_365_label_map[cls]
                cls_count[gcls] = 0

        for ds in ["train", "val"]:
            with open(os.path.join(self.data_dir, self.places_image_lists[ds])) as f:
                for path, cls_str in [l.strip().split(" ", 1) for l in f.readlines()]:
                    cls = int(cls_str)
                    gcls = places_365_gcls_map[cls]  # Add label
                    img = os.path.join(self.places_dirs[ds], path.lstrip("/"))
                    insert_img(cls_imgs, gcls, img)
                    cls_count[gcls] += 1

        # Prepare Places 69 extra
        with open(os.path.join(self.data_dir, self.places_label_files["69"])) as f:
            places_69_label_map = {int(cls): cls_str for cls_str, cls in
                                   [l.strip().split(" ", 1) for l in f.readlines()]}
            num_places_69_cls = len(places_69_label_map)
            places_69_gcls_map = {cls: cls + current_cls for cls in range(num_places_69_cls)}
            current_cls += num_places_69_cls
            for cls, gcls in places_69_gcls_map.items():
                cls_map[gcls] = places_69_label_map[cls]
                cls_count[gcls] = 0

        with open(os.path.join(self.data_dir, self.places_image_lists["extra"])) as f:
            for path, cls_str in [l.strip().split(" ", 1) for l in f.readlines()]:
                cls = int(cls_str)
                gcls = places_69_gcls_map[cls]  # Add label
                img = os.path.join(self.places_dirs["extra"], path.lstrip("/"))
                insert_img(cls_imgs, gcls, img)
                cls_count[gcls] += 1

        # Remove last 35 because they have < 1000 images
        for k in range(len(cls_map) - 35, len(cls_map)):
            del cls_imgs[k]
            del cls_map[k]
            del cls_count[k]

        log("done.")

        return cls_imgs, cls_map, cls_count


if __name__ == '__main__':
    d = ImageNetPlacesBuilder()

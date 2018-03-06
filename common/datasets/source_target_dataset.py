import os
import glob
import cv2
from .datasets_base import datasets_base
import numpy as np

class source_target_dataset(datasets_base):
    def __init__(self, dataset_a, dataset_b, flip=1, resize_to=280, crop_to=256):
        if os.path.isdir(dataset_a):
            self.train_a_key = []
            self.train_a_key.extend(glob.glob(os.path.join(dataset_a, "*.jpg")))
            self.train_a_key.extend(glob.glob(os.path.join(dataset_a, "*.png")))
            self.train_a_key.extend(glob.glob(os.path.join(dataset_a, "*.tif")))
            if len(self.train_a_key) == 0:
                raise Exception("No .jpg or .png or .tif file in " + dataset_a)
        if os.path.isdir(dataset_b):
            self.train_b_key = []
            self.train_b_key.extend(glob.glob(os.path.join(dataset_b, "*.jpg")))
            self.train_b_key.extend(glob.glob(os.path.join(dataset_b, "*.png")))
            self.train_b_key.extend(glob.glob(os.path.join(dataset_b, "*.tif")))
            if len(self.train_b_key) == 0:
                raise Exception("No .jpg or .png or .tif file in " + dataset_b)
            import random
            random.shuffle(self.train_b_key)
        super(source_target_dataset, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to, keep_aspect_ratio=True)
        self.epoch = 0

    def __len__(self):
        return min(len(self.train_a_key), len(self.train_b_key))

    def get_example(self, i):
        idA = self.train_a_key[i % len(self.train_a_key)]
        current_epoch = i // len(self.train_b_key)
        if current_epoch > self.epoch:
            self.epoch = current_epoch
            import random
            random.shuffle(self.train_b_key)
        idB = self.train_b_key[i%len(self.train_b_key)]

        imgA = cv2.imread(idA, cv2.IMREAD_COLOR)
        imgB = cv2.imread(idB, cv2.IMREAD_COLOR)
        #convert BGR to RGB for chainercv fashion
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

        # imgA = self.do_augmentation(imgA)
        # imgB = self.do_augmentation(imgB)

        imgA = self.preprocess_image(imgA)
        imgB = self.preprocess_image(imgB)

        idA_ , ext = os.path.splitext(idA)
        annotation_file = idA_ + ".txt"

        imgA_gt_map = np.zeros(imgA.shape).astype("f")

        bbox = []
        label = []

        with open(annotation_file, "r") as annotations:
            line = annotations.readline()
            while (line):
                xmin, ymin, xmax, ymax = line.split(",")
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                imgA_gt_map[...,ymin-1:ymax,xmin-1:xmax] =1
                bbox.append([ymin - 1, xmin - 1, ymax - 1, xmax - 1])  # obey the rule of chainercv
                label.append(0) #class number
                line = annotations.readline()

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        return imgA, imgA_gt_map, imgB, bbox, label

    @property
    def len_A(self):
        return len(self.train_a_key)

    @property
    def len_B(self):
        return len(self.train_b_key)

    def get_example_raw_A(self,i):
        idA = self.train_a_key[i % len(self.train_a_key)]
        imgA = cv2.imread(idA, cv2.IMREAD_COLOR)
        # convert BGR to RGB for chainercv fashion
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgA = self.preprocess_image(imgA)
        return imgA

    def get_example_raw_B(self,i):
        idB = self.train_b_key[i % len(self.train_b_key)]
        imgB = cv2.imread(idB, cv2.IMREAD_COLOR)
        # convert BGR to RGB for chainercv fashion
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
        imgB = self.preprocess_image(imgB)
        return imgB

    def get_filename_A(self,i):
        return self.train_a_key[i % len(self.train_a_key)]
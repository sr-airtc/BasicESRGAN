# -*- coding:utf-8 -*-
from tensorflow.python import keras
import math
# from augmentor import *
import cv2
import glob
import os
from PIL import Image
import random
import numpy as np
from tensorflow.python.keras.utils import to_categorical

transpose = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
             Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

class dataGenertor(keras.utils.Sequence):
    def __init__(self, directory, batch_size=1, expansion=1, crop_w=162, crop_h=279, return_labels=False):
        self.batch_size = batch_size
        self.filenames = self.get_filenames(directory)
        self.length = len(self.filenames)
        self.crop_w = expansion*crop_w
        self.crop_h = expansion*crop_h
        self.return_labels = return_labels

    def get_filenames(self, directory):
        self.filenames = []
        self.filelabels = []
        files = glob.glob(os.path.join(directory, '*'))
        print('%s has %d examples' % (directory, len(files)))
        return files
    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        try:
            data = [self.process_image(self.filenames[index+batch]) for batch in range(self.batch_size)]
        except IOError:
            print([self.filenames[index+batch] for batch in range(self.batch_size)])
            raise IOError('ERROR!!!!!!!!!!!')
        if self.return_labels:
            return np.array(data), np.ones((self.batch_size, 1))
        else:
            return np.array(data)

    def process_image(self, image):
        """Given an image, process it and return the array."""
        img = Image.open(image).convert("RGB")
        img = np.array(img)
        img = random_crop(img, self.crop_w, self.crop_h)
        img = img.astype(np.float32)/127.5 - 1
        return img


class dataDiscriminator(keras.utils.Sequence):
    def __init__(self, lr_dir, hr_dir, model, batch_size=16, expansion=4, crop_w=162, crop_h=279):
        self.batch_size = batch_size
        self.lr_filename, self.hr_filename = self.get_filenames(lr_dir, hr_dir)
        self.length = len(self.lr_filename)
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.expansion = expansion
        self.model = model

    def get_filenames(self, lr_dir, hr_dir):
        lr_filename = glob.glob(os.path.join(lr_dir, '*'))
        hr_filename = glob.glob(os.path.join(hr_dir, '*'))
        if len(lr_filename) != len(hr_filename):
            raise ValueError('高清数据数量%d与低清数据数量%d不相等' % (len(hr_filename), len(lr_filename)))
        return lr_filename, hr_filename

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(self.length / float(self.batch_size))

    def __getitem__(self, index):
        imgs = []
        try:
            for batch in range(self.batch_size):
                number = index+batch
                lr_img = self.process_image(self.lr_filename[number])
                sr_img = self.model.predict(np.expand_dims(lr_img, axis=0))[0]
                imgs.append(sr_img)
                hr_img = self.process_image(self.hr_filename[number], expansion=4)
                imgs.append(hr_img)
        except IOError:
            print(self.lr_filename[number], self.hr_filename[number])
            raise IOError('ERROR!!!!!!!!!!!')
        labels = np.array([0, 1]*self.batch_size)
        return np.array(imgs), np.expand_dims(labels, axis=-1)

    def process_image(self, image, expansion=1):
        """Given an image, process it and return the array."""
        img = Image.open(image).convert("RGB")
        img = np.array(img)
        img = random_crop(img, expansion*self.crop_w, expansion*self.crop_h)
        img = img.astype(np.float32)/127.5 - 1
        return img


def random_crop(img, crop_w=162, crop_h=279):
    w, h, c = img.shape
    shift_w = random.randint(0, w - crop_w)
    shift_h = random.randint(0, h -crop_h)
    img = img[shift_w:shift_w+crop_w, shift_h:shift_h+crop_h, :]
    return img
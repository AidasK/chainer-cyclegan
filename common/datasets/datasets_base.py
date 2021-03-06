import cv2
from chainer.dataset import dataset_mixin
import numpy as np

class datasets_base(dataset_mixin.DatasetMixin):
    def __init__(self, flip=1, resize_to=64, crop_to=0, random_brightness=0, keep_aspect_ratio=False):
        self.flip = flip
        self.resize_to = resize_to
        self.crop_to  = crop_to
        self.random_brightness = random_brightness
        self.keep_aspect_ratio = keep_aspect_ratio

    def preprocess_image(self, img):
        img = img.astype("f")
        img = img / 255
        img = (img - 0.5) / 0.5
        img = img.transpose((2, 0, 1))
        return img

    def do_random_crop(self, img, crop_to):
        sz0 = img.shape[0]
        sz1 = img.shape[1]
        if sz0 > crop_to and sz1 > crop_to:
            lim0 = sz0 - crop_to
            lim1 = sz1 - crop_to
            x = np.random.randint(0,lim0)
            y = np.random.randint(0,lim1)
            img = img[x:x+crop_to, y:y+crop_to]
        return img

    def do_resize(self, img, resize_to, keep_aspect_ratio=False):
        if isinstance(resize_to, tuple):
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        elif not keep_aspect_ratio:
            img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
        else:
            h, w, c = img.shape
            if h >= w:
                nw = resize_to
                nh = nw * h // w
            else:
                nh = resize_to
                nw = nh * w // h
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return img

    def do_flip(self, img):
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
        return img

    def do_random_brightness(self, img):
        if np.random.rand() > 0.7:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int)
        hsv[:,:,2] += np.random.randint(-40,70)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def do_augmentation(self, img):
        if self.flip > 0:
            img = self.do_flip(img)

        if self.random_brightness > 0:
            img = self.do_random_brightness(img)

        if self.resize_to > 0:
            img = self.do_resize(img, self.resize_to, self.keep_aspect_ratio)

        if self.crop_to > 0:
            img = self.do_random_crop(img, self.crop_to)

        return img

import cv2
import numpy as np
from chainer import cuda
import chainer
try:
    import cupy
except:
    pass

def copy_to_cpu(imgs):
    if type(imgs) == chainer.variable.Variable :
        imgs = imgs.data
    try:
        if type(imgs) == cupy.core.core.ndarray:
            imgs = cuda.to_cpu(imgs)
    except:
        pass
    return imgs

def postprocessing_tanh(imgs):
    imgs = (imgs * 0.5 + 0.5) * 255
    imgs = imgs.astype(np.uint8)
    return imgs

def save_single_image(img, path, post_processing=postprocessing_tanh):
    img = copy_to_cpu(img)
    if post_processing is not None:
        img = post_processing(img)
    #ch, w, h = img.shape
    img = img.transpose((1, 2, 0))
    #for chainercv fashion
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def save_images_grid(imgs, path, post_processing=postprocessing_tanh, transposed=False):
    imgs = copy_to_cpu(imgs)
    if post_processing is not None:
        imgs = post_processing(imgs)
    b, ch, w, h = imgs.shape
    batch = int(b / 2 / 3)

    imgs = imgs.reshape(2,3,batch, ch, w,h)
    imgs = imgs.transpose(0,2,4,1,5,3)
    imgs = imgs.reshape(2,batch,w,3*h,ch)
    imgs = imgs.reshape(2,batch*w,3*h,ch)
    imgs = imgs.reshape(2*batch*w,3*h,ch)

    if ch == 1:
        imgs.reshape(2 * batch * w, 3 * h)
    # for chainercv fashion
    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, imgs)


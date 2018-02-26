import numpy as np
import chainer.functions as F

def loss_l1(h, t):
    return F.sum(F.absolute(h-t)) / np.prod(h.data.shape)

def loss_l2(h, t):
    return F.sum((h-t)**2) / np.prod(h.data.shape)

def loss_l2_norm(h, t, axis=(1)):
    return F.sum(F.sqrt(F.sum((h-t)**2, axis=axis))) / h.data.shape[0]

def loss_func_dcgan_dis_real(y_real):
    return F.sum(F.softplus(-y_real)) / np.prod(y_real.data.shape)

def loss_func_dcgan_dis_fake(y_fake):
    return F.sum(F.softplus(y_fake)) / np.prod(y_fake.data.shape)

def loss_func_lsgan_dis_real(y_real, label=1.0):
    return loss_l2(y_real, label)

def loss_func_lsgan_dis_fake(y_fake, label=0.0):
    return loss_l2(y_fake, label)

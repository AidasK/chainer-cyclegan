import functools

import chainer
import chainer.functions as F
import chainer.links as L

# from instance_normalization import InstanceNormalization
from common.instance_norm_v2 import InstanceNormalization

def reflectPad(x, pad):
    if pad < 0:
        print("Pad width has to be 0 or larger")
        raise ValueError
    if pad == 0:
        return x
    else:
        width, height = x.shape[2:]
        w_pad = h_pad = pad
        if width == 1:
            x = F.concat((x,)*(1+pad*2),axis=2)
        else:
            while w_pad > 0:
                pad = min(w_pad, width-1)
                w_pad -= pad
                x = _pad_along_axis(x, pad, 2)
                width, height = x.shape[2:]
        if height == 1:
            x = F.concat((x,)*(1+pad*2),axis=3)
        else:
            while h_pad > 0:
                pad = min(h_pad, height-1)
                h_pad -= pad
                x = _pad_along_axis(x, pad, 3)
                width, height = x.shape[2:]
        return x

def _pad_along_axis(x, pad, axis):
    dim = x.ndim
    head = F.get_item(x,(slice(None),) * axis + (slice(1, 1 + pad),) + (slice(None),)*(dim-1-axis))
    head = F.concat(
        [F.get_item(head, (slice(None),) * axis + (slice(i, i + 1),) + (slice(None),)*(dim-1-axis)) for i in range(pad)][::-1], axis=axis)
    tail = F.get_item(x, (slice(None),) * axis + (slice(-1-pad, -1),) + (slice(None),)*(dim-1-axis))
    tail = F.concat(
        [F.get_item(tail, (slice(None),) * axis + (slice(i, i + 1),) + (slice(None),)*(dim-1-axis)) for i in range(pad)][::-1],
        axis=axis)
    x = F.concat((head, x, tail), axis=axis)
    return x

def get_norm_layer(norm='instance'):
    # unchecked: init weight of bn
    if norm == 'batch':
        norm_layer = functools.partial(L.BatchNormalization, use_gamma=True,
                                       use_beta=True)
    elif norm == 'instance':
        norm_layer = functools.partial(InstanceNormalization, use_gamma=True,
                                       use_beta=True)
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm)
    return norm_layer


class ResBlock(chainer.Chain):
    def __init__(self, ch, norm='instance', pad_type='reflect', activation=F.relu):
        super(ResBlock, self).__init__()
        self.activation = activation
        w = chainer.initializers.Normal(0.02)
        if pad_type == 'reflect':
            pad = 0
        elif pad_type == 'zero':
            pad = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % pad_type)

        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, pad, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, pad, initialW=w)
            self.norm0 = get_norm_layer(norm)(ch)
            self.norm1 = get_norm_layer(norm)(ch)
            self.pad_type = pad_type

    def __call__(self, x):
        if self.pad_type == 'reflect':
            h = reflectPad(x,1)
            h = self.c0(h)
        else:
            h = self.c0(x)
        h = self.norm0(h)
        h = self.activation(h)
        if self.pad_type == 'reflect':
            h = reflectPad(h,1)
        h = self.c1(h)
        h = self.norm1(h)
        return h + x

class CNA(chainer.Chain):
    def __init__(self, ch0, ch1, ksize=3, pad=1, norm='instance',
                 sample='down', activation=F.relu, dropout=False):
        super(CNA, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        w = chainer.initializers.Normal(0.02)
        self.use_norm = False if norm is None else True

        with self.init_scope():
            if sample == 'down':
                self.c = L.Convolution2D(ch0, ch1, ksize, 2, pad, initialW=w)
            elif sample == 'none-9':
                self.c = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w)
            elif sample == 'none-7':
                self.c = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w)
            elif sample == 'none-7_nopad':
                self.c = L.Convolution2D(ch0, ch1, 7, 1, 0, initialW=w)
            elif sample == 'none-5':
                self.c = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
            elif sample == 'up':
                self.c = L.Deconvolution2D(ch0,ch1,3,2,1,initialW=w)
            else:
                self.c = L.Convolution2D(ch0, ch1, ksize, 1, pad, initialW=w)
            if self.use_norm:
                self.norm = get_norm_layer(norm)(ch1)

    def __call__(self, x):
        # if self.sample in ['down', 'none', 'none-9', 'none-7','none-7_nopad', 'none-5']:
        #     h = self.c(x)
        if self.sample == 'up':
            h = F.pad(x,((0,0),(0,0),(0,1),(0,1)),mode='constant')
            h = self.c(h)
            h = F.get_item(h,(slice(None),slice(None),slice(0,-1),slice(0,-1)))
        else:
            h = self.c(x)
        if self.use_norm:
            h = self.norm(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class Generator(chainer.Chain):
    def __init__(self, norm='instance', n_resblock=9,pad_type = 'reflect'):
        super(Generator, self).__init__()
        self.n_resblock = n_resblock
        with self.init_scope():
            # nn.ReflectionPad2d in original
            if pad_type == 'reflect':
                self.c1 = CNA(3, 32, norm=norm, sample='none-7_nopad')
            else:
                self.c1 = CNA(3, 32, norm=norm, sample='none-7')
            self.c2 = CNA(32, 64, norm=norm, sample='down')
            self.c3 = CNA(64, 128, norm=norm, sample='down')
            for i in range(n_resblock):
                setattr(self, 'c' + str(i + 4), ResBlock(128, norm=norm))
            # nn.ConvTranspose2d in original
            setattr(self, 'c' + str(n_resblock + 4),
                    CNA(128, 64, norm=norm, sample='up'))
            setattr(self, 'c' + str(n_resblock + 5),
                    CNA(64, 32, norm=norm, sample='up'))
            if pad_type == 'reflect':
                setattr(self, 'c' + str(n_resblock + 6),
                        CNA(32, 3, norm=norm, sample='none-7_nopad', activation=F.tanh))
            else:
                setattr(self, 'c' + str(n_resblock + 6),
                        CNA(32, 3, norm=norm, sample='none-7', activation=F.tanh))
            self.pad_type= pad_type

    def __call__(self, x):
        if self.pad_type == 'reflect':
            h = reflectPad(x,3)
            h = self.c1(h)
        else:
            h = self.c1(x)
        for i in range(2, self.n_resblock + 6):
            h = getattr(self, 'c' + str(i))(h)
        if self.pad_type == 'reflect':
            h = reflectPad(h,3)
        h = getattr(self, 'c' + str(self.n_resblock + 6))(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, norm='instance', in_ch=3, n_down_layers=3):
        super(Discriminator, self).__init__()
        base = 64
        ksize = 4
        pad = 1
        self.n_down_layers = n_down_layers

        with self.init_scope():
            self.c0 = CNA(in_ch, 64, ksize=ksize, pad=pad, norm=None,
                          sample='down', activation=F.leaky_relu,
                          dropout=False)

            for i in range(1, n_down_layers):
                setattr(self, 'c' + str(i),
                        CNA(base, base * 2, ksize=ksize, pad=pad, norm=norm,
                            sample='down', activation=F.leaky_relu,
                            dropout=False))
                base *= 2

            setattr(self, 'c' + str(n_down_layers),
                    CNA(base, base * 2, ksize=ksize, pad=pad, norm=norm,
                        sample='none', activation=F.leaky_relu, dropout=False))
            base *= 2

            setattr(self, 'c' + str(n_down_layers + 1),
                    CNA(base, 1, ksize=ksize, pad=pad, norm=None,
                        sample='none', activation=None, dropout=False))

    def __call__(self, x_0):
        h = self.c0(x_0)
        for i in range(1, self.n_down_layers + 2):
            h = getattr(self, 'c' + str(i))(h)
        return h

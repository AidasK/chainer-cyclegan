import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .backwards import *
from common.instance_norm_v2 import InstanceNormalization

def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if not chainer.config.train:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

def weight_clipping(model, lower=-0.01, upper=0.01):
    for params in model.params():
        params_clipped = F.clip(params, lower, upper)
        params.data = params_clipped.data

class ResBlock(chainer.Chain):
    def __init__(self, ch, norm=None, activation=F.relu, k_size=3, w_init=None, reflect = 0, norm_learnable = True, normalize_grad = False):
        if w_init == None:
            w = chainer.initializers.HeNormal()
        else:
            w = w_init
        if norm in ['instance','bn']:
            use_norm = True
        else:
            use_norm = False
        self.use_norm = use_norm
        self.activation = activation
        layers = {}

        #self.nopadding = nopadding
        self.reflect = reflect

        if self.reflect in [1,2]:
            pad = 0
        else:
            pad = k_size//2

        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, pad, initialW=w)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, pad, initialW=w)

        # if isinstance(bn, tuple) and bn[0] == 'multi_node_bn':
        #     import chainermn
        #     comm = bn[1]
        #     layers['bn0'] = chainermn.links.MultiNodeBatchNormalization(ch, comm)
        #     layers['bn1'] = chainermn.links.MultiNodeBatchNormalization(ch, comm)
        # elif bn:  # works if bn == True or bn == 'bn'
        #     layers['bn0'] = L.BatchNormalization(ch)
        #     layers['bn1'] = L.BatchNormalization(ch)

        if self.use_norm:
            if norm == 'instance':
                layers['norm0'] = InstanceNormalization(ch, use_gamma=norm_learnable, use_beta=norm_learnable, norm_grad=normalize_grad)
                layers['norm1'] = InstanceNormalization(ch, use_gamma=norm_learnable, use_beta=norm_learnable, norm_grad=normalize_grad)
            elif norm == 'bn':
                layers['norm0'] = L.BatchNormalization(ch, use_gamma=norm_learnable, use_beta=norm_learnable)
                layers['norm1'] = L.BatchNormalization(ch, use_gamma=norm_learnable, use_beta=norm_learnable)

        super(ResBlock, self).__init__(**layers)
        self.register_persistent('reflect')
        self.register_persistent('use_norm')
        self.register_persistent('activation')

    #override serizalize to support serializing function object
    def serialize(self, serializer):
        """Serializes the link object.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        d = self.__dict__
        for name in self._params:
            param = d[name]
            data = serializer(name, param.data)
            if param.data is None and data is not None:
                # Initialize the parameter here
                param.initialize(data.shape)
                if isinstance(param.data, np.ndarray):
                    np.copyto(param.data, data)
                else:
                    param.data.set(np.asarray(data))
        for name in self._persistent:
            if isinstance(serializer,chainer.serializer.Deserializer) and name == "activation":
                d[name] = None
            d[name] = serializer(name, d[name])
            if isinstance(serializer, chainer.serializer.Deserializer) and name == "activation":
                if isinstance(d[name],np.ndarray):
                    d[name] = d[name][()]

    def __call__(self, x):
        if self.reflect == 2:
            h = F.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect')
        else:
            h = x
        h = self.c0(h)
        if self.use_norm:
            h = self.norm0(h)
        h = self.activation(h)
        if self.reflect == 2:
            h = F.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect')
        h = self.c1(h)
        if self.use_norm:
            h = self.norm1(h)
        if self.reflect == 1:
            x = F.get_item(x,(slice(None),slice(None),slice(2,-2),slice(2,-2)))
        return h + x

class DownResBlock(chainer.Chain):
    """
        pre activation residual block
    """
    def __init__(self, ch, normal_init=0.02):
        w = chainer.initializers.Normal(normal_init)
        super(DownResBlock, self).__init__(
            c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w),
            c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w),
            c2 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w),
            c3 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w),
            c4 = L.Convolution2D(ch, ch*2, 4, 2, 1, initialW=w)
        )

    def __call__(self, x):
        self.h0 = x
        self.h1 = self.c0(F.leaky_relu(self.h0))
        self.h2 = self.c1(F.leaky_relu(self.h1))
        self.h3 = self.h2 + self.h0
        self.h4 = self.c2(F.leaky_relu(self.h3))
        self.h5 = self.c3(F.leaky_relu(self.h4))
        self.h6 = self.h5 + self.h3
        self.h7 = self.c4(F.leaky_relu(self.h6))
        return self.h7

    def differentiable_backward(self, g):
        g = backward_convolution(self.h6, g, self.c4)
        g_ = g = backward_leaky_relu(self.h6, g, 0.2)
        g = backward_convolution(self.h4, g, self.c3)
        g = backward_leaky_relu(self.h4, g, 0.2)
        g = backward_convolution(self.h3, g, self.c2)
        g = backward_leaky_relu(self.h3, g, 0.2)
        g_ = g = g + g_
        g = backward_convolution(self.h1, g, self.c1)
        g = backward_leaky_relu(self.h1, g, 0.2)
        g = backward_convolution(self.h0, g, self.c0)
        g = backward_leaky_relu(self.h0, g, 0.2)
        g = g + g_
        return g

class NNBlock(chainer.Chain):
    def __init__(self, ch0, ch1, \
                nn='conv', \
                norm=None, \
                activation=F.relu, \
                dropout=False, \
                noise=None, \
                w_init=None, \
                k_size = 3, \
                pad = None, \
                normalize_input=False,\
                norm_learnable = True,\
                normalize_grad = False):

        self.norm = norm
        self.normalize_input = normalize_input
        self.activation = activation
        self.dropout = dropout
        self.noise = noise
        self.nn = nn
        self.norm_learnable = norm_learnable
        self.normalize_grad = normalize_grad
        layers = {}

        if w_init == None:
            w = chainer.initializers.GlorotNormal()
        else:
            w = w_init

        if nn == 'down_conv':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)

        elif nn == 'down_conv_2':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 1, 1, initialW=w)

        elif nn == 'g_down_conv':
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 2, 1, initialW=w)

        elif nn == 'up_deconv':
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)

        elif nn == 'up_subpixel':
            pad = k_size//2
            layers['c'] = L.Convolution2D(ch0, ch1*4, k_size, 1, pad, initialW=w)

        elif nn=='conv' or nn=='up_unpooling':
            if pad == None:
                pad = k_size//2
            layers['c'] = L.Convolution2D(ch0, ch1, k_size, 1, pad, initialW=w)

        elif nn=='linear':
            layers['c'] = L.Linear(ch0, ch1, initialW=w)

        else:
            raise Exception("Cannot find method %s" % nn)

        if self.norm == 'bn':
            if self.noise:
                layers['n'] = L.BatchNormalization(ch1, use_gamma=False)
            else:
                layers['n'] = L.BatchNormalization(ch1)
        elif self.norm == 'instance':
            if self.noise:
                layers['n'] = InstanceNormalization(ch1, use_gamma=False, use_beta=self.norm_learnable,\
                                                    norm_grad = self.normalize_grad)
            else:
                layers['n'] = InstanceNormalization(ch1, use_gamma=self.norm_learnable, use_beta=self.norm_learnable,\
                                                    norm_grad = self.normalize_grad)
        elif isinstance(self.norm, tuple) and self.norm[0] == 'multi_node_bn':
            import chainermn
            comm = self.norm[1]
            if self.noise:
                layers['n'] = chainermn.links.MultiNodeBatchNormalization(ch1, comm, use_gamma=False)
            else:
                layers['n'] = chainermn.links.MultiNodeBatchNormalization(ch1, comm)
        elif self.norm == 'ln':
                layers['n'] = L.LayerNormalization(ch1)

        super(NNBlock, self).__init__(**layers)
        self.register_persistent('normalize_input')
        self.register_persistent('norm')
        self.register_persistent('activation')
        self.register_persistent('nn')
        self.register_persistent('noise')
        self.register_persistent('dropout')

    def _do_normalization(self, x, retain_forward=False):
        if str(self.norm) in ['bn', 'instance','multi_node_bn']:
            return self.n(x)
        elif str(self.norm) == 'ln':
            y = self.n(x)
            if retain_forward:
                self.nx = y
            return y
        else:
            return x

    def _do_before_cal(self, x):
        if self.nn == 'up_unpooling':
            x = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
        return x

    def _do_after_cal_0(self, x):
        if self.nn == 'up_subpixel':
            x = F.depth2space(x, 2)
        return x

    def _do_after_cal_1(self, x):
        if self.noise:
            x = add_noise(x)
        if self.dropout:
            x = F.dropout(x)
        return x

    def serialize(self, serializer):
        """Serializes the link object.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        d = self.__dict__
        for name in self._params:
            param = d[name]
            data = serializer(name, param.data)
            if param.data is None and data is not None:
                # Initialize the parameter here
                param.initialize(data.shape)
                if isinstance(param.data, np.ndarray):
                    np.copyto(param.data, data)
                else:
                    param.data.set(np.asarray(data))
        for name in self._persistent:
            if isinstance(serializer,chainer.serializer.Deserializer) and name == "activation":
                d[name] = None
            d[name] = serializer(name, d[name])
            if isinstance(serializer, chainer.serializer.Deserializer) and name == "activation":
                if isinstance(d[name],np.ndarray):
                    d[name] = d[name][()]

    def __call__(self, x, retain_forward=False):
        if self.normalize_input:
            x = self._do_normalization(x, retain_forward=retain_forward)

        x = self._do_before_cal(x)
        x = self.c(x)
        x = self._do_after_cal_0(x)
        if self.norm != None and not self.normalize_input:
            x = self._do_normalization(x, retain_forward=retain_forward)
        x = self._do_after_cal_1(x)

        if self.activation != None:
            x = self.activation(x)

        if retain_forward:
            self.x = x
        return x

    def differentiable_backward(self, g):
        if self.normalize_input:
            raise NotImplementedError

        if self.activation is F.leaky_relu:
            g = backward_leaky_relu(self.x, g)
        elif self.activation is F.relu:
            g = backward_relu(self.x, g)
        elif self.activation is F.tanh:
            g = backward_tanh(self.x, g)
        elif self.activation is F.sigmoid:
            g = backward_sigmoid(self.x, g)
        elif not self.activation is None:
            raise NotImplementedError

        if self.norm == 'ln':
            g = backward_layernormalization(self.nx, g, self.n)
        elif not self.norm is None:
            raise NotImplementedError

        if self.nn == 'down_conv' or self.nn == 'conv':
            g = backward_convolution(None, g, self.c)
        elif self.nn == 'linear':
            g = backward_linear(None, g, self.c)
        elif self.nn == 'up_deconv':
            g = backward_deconvolution(None, g, self.c)
        else:
            raise NotImplementedError

        return g

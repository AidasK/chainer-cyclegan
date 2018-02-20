import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .ops import *
from common.instance_norm_v2 import InstanceNormalization

class ResNetImageTransformer(chainer.Chain):
    def __init__(self, base_channels=32, norm_func='instance', init_std=0.02, down_layers=2,
                 res_layers=9, up_layers=2, upsampling='up_unpooling', \
                 reflect=0, norm_learnable=True, normalize_grad=False):
        layers = {}
        self.down_layers = down_layers
        self.res_layers = res_layers
        self.up_layers = up_layers
        self.base_channels = base_channels
        self.reflect = reflect
        #nopadding = reflect

        # if isinstance(use_bn, tuple) and use_bn[0] == 'multi_node_bn':
        #     norm = use_bn
        #     w = chainer.initializers.Normal(normal_init)
        # elif use_bn:
        #     norm = 'bn'
        #     w = chainer.initializers.Normal(normal_init)
        # else:
        #     norm = None
        #     w = None
        if norm_func == 'instance':
            norm = 'instance'
        elif norm_func == 'bn':
            norm = 'bn'
        else:
            norm = None
        w = chainer.initializers.Normal(init_std)

        base = base_channels
        if reflect == 2:
            layers['c_first'] = NNBlock(3, base, nn='conv', k_size=7, norm=norm, w_init=w, pad=0,\
                                        norm_learnable=norm_learnable, normalize_grad=normalize_grad)
        else:
            layers['c_first'] = NNBlock(3, base, nn='conv', k_size=7, norm=norm, w_init=w, \
                                         norm_learnable=norm_learnable, normalize_grad=normalize_grad)
        for i in range(self.down_layers):
            layers['c_down_'+str(i)] = NNBlock(base, base*2, nn='g_down_conv', norm=norm, w_init=w, \
                                               norm_learnable = norm_learnable, normalize_grad = normalize_grad)
            base = base * 2
        for i in range(self.res_layers):
            layers['c_res_'+str(i)] = ResBlock(base, norm=norm, w_init=w, reflect=reflect, norm_learnable=norm_learnable,\
                                               normalize_grad=normalize_grad)
        for i in range(self.up_layers):
            layers['c_up_'+str(i)] = NNBlock(base, base//2, nn=upsampling, norm=norm, w_init=w, \
                                             norm_learnable=norm_learnable, normalize_grad=normalize_grad)
            base = base // 2
        if reflect == 2:
            layers['c_last'] = NNBlock(base, 3, nn='conv', k_size=7, norm=None, w_init=w, pad=0, activation=F.tanh, \
                                       norm_learnable=norm_learnable, normalize_grad=normalize_grad)
        else:
            layers['c_last'] = NNBlock(base, 3, nn='conv', k_size=7, norm=None, w_init=w, activation=F.tanh, \
                                        norm_learnable=norm_learnable, normalize_grad=normalize_grad)

        super(ResNetImageTransformer, self).__init__(**layers)
        self.register_persistent('reflect')
        self.register_persistent('down_layers')
        self.register_persistent('res_layers')
        self.register_persistent('up_layers')

    def __call__(self, x):
        if self.reflect == 2:
            self.c_first.c.pad = (0, 0)
            self.c_last.c.pad = (0, 0)
        else:
            _pad_f = self.c_first.c.W.shape[2] // 2
            _pad_l = self.c_last.c.W.shape[2] // 2
            self.c_first.c.pad = (_pad_f, _pad_f)
            self.c_last.c.pad = (_pad_l, _pad_l)
        if self.reflect == 1:
            reflect_pad = 2**self.down_layers * 4 * self.res_layers // 2
            # x = F.pad(x,((0,0),(0,0),(reflect_pad,reflect_pad),(reflect_pad,reflect_pad)),mode='reflect')
            x = reflectPad(x, reflect_pad)
        elif self.reflect == 2:
            # x = F.pad(x, ((0, 0), (0, 0), (3, 3), (3, 3)), mode='reflect')
            x = reflectPad(x, 3)
        # _pair = self.c_fisrt._pair
        # self.c_first.pad = _pair(3) if self.reflect == 0 else _pair(0)
        h = self.c_first(x)
        for i in range(self.down_layers):
            h = getattr(self, 'c_down_'+str(i))(h)
        for i in range(self.res_layers):
            h = getattr(self, 'c_res_'+str(i))(h)
        for i in range(self.up_layers):
            h = getattr(self, 'c_up_'+str(i))(h)
        if self.reflect == 2:
            # h = F.pad(h, ((0, 0), (0, 0), (3, 3), (3, 3)), mode='reflect')
            h = reflectPad(h, 3)
        # self.c_last.pad = _pair(3) if self.reflect == 0 else _pair(0)
        h = self.c_last(h)
        return h

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import six
from chainer import cuda, optimizers, serializers, Variable
import sys
sys.path.insert(0, '../')
from common.loss_functions import *
from common.models.backwards import *

class HistoricalBuffer():
    def __init__(self, xp, buffer_size=50, image_size=256, image_channels=3):
        self._buffer_size = buffer_size
        self._img_size = image_size
        self._img_ch = image_channels
        self._cnt = 0
        self._buffer = xp.zeros((self._buffer_size, self._img_ch, self._img_size, self._img_size)).astype("f")

    def get_and_update(self, data, prob=0.5):
        #print(self._buffer.shape, data.shape)
        if self._cnt < self._buffer_size:
            self._buffer[self._cnt, :] = data
            self._cnt += 1
            return data
        pos = self._cnt % self._buffer_size
        #print(pos, self._buffer.shape, data.shape)
        self._buffer[pos, : ]=data
        if np.random.rand() < prob:
            return data
        id = np.random.randint(0, self._buffer_size)
        return self._buffer[id, :].reshape((1, self._img_ch, self._img_size, self._img_size))

    def serialize(self, serializer):
        self._cnt = serializer('cnt', self._cnt)
        self._buffer = serializer('buffer', self._buffer)

class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        self._iter = 0
        params = kwargs.pop('params')
        self._lambda1 = params['lambda1']
        self._lambda2 = params['lambda2']
        self._learning_rate_anneal = params['learning_rate_anneal']
        self._learning_rate_anneal_interval = params['learning_rate_anneal_interval']
        self._learning_rate_anneal_trigger = params['learning_rate_anneal_trigger']
        self._image_size = params['image_size']
        self._max_buffer_size = params['buffer_size']

        xp = self.gen_g.xp
        self._buffer_x = HistoricalBuffer(xp, self._max_buffer_size, self._image_size)
        self._buffer_y = HistoricalBuffer(xp, self._max_buffer_size, self._image_size)
        super(Updater, self).__init__(*args, **kwargs)


    def update_core(self):
        xp = self.gen_g.xp
        self._iter += 1
        batch = self.get_iterator('main').next()

        batchsize = len(batch)

        w_in = self._image_size

        x = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")
        y = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")

        #print(batchsize)
        #print(x.shape)
        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            y[i, :] = xp.asarray(batch[i][1])

        x = Variable(x)
        y = Variable(y)

        x_y = self.gen_g(x)
        x_y_copy = self._buffer_x.get_and_update(x_y.data)
        x_y_copy = Variable(x_y_copy)
        x_y_x = self.gen_f(x_y)

        y_x = self.gen_f(y)
        y_x_copy = self._buffer_x.get_and_update(y_x.data)
        y_x_copy = Variable(y_x_copy)
        y_x_y = self.gen_g(y_x)

        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')

        self.dis_y.cleargrads()
        self.dis_x.cleargrads()

        loss_dis_y_fake = loss_func_lsgan_dis_fake(self.dis_y(x_y_copy))
        loss_dis_y_real = loss_func_lsgan_dis_real(self.dis_y(y))
        loss_dis_y = loss_dis_y_fake + loss_dis_y_real
        chainer.report({'loss': loss_dis_y}, self.dis_y)

        loss_dis_x_fake = loss_func_lsgan_dis_fake(self.dis_x(y_x_copy))
        loss_dis_x_real = loss_func_lsgan_dis_real(self.dis_x(x))
        loss_dis_x = loss_dis_x_fake + loss_dis_x_real
        chainer.report({'loss': loss_dis_x}, self.dis_x)

        loss_dis_y.backward()
        loss_dis_x.backward()

        opt_y.update()
        opt_x.update()

        self.gen_f.cleargrads()
        self.gen_g.cleargrads()

        loss_gen_g_adv = self._lambda2 * loss_func_lsgan_dis_real(self.dis_y(x_y))
        loss_gen_f_adv = self._lambda2 * loss_func_lsgan_dis_real(self.dis_x(y_x))
        loss_cycle_x = self._lambda1 * loss_l1(x_y_x, x)
        loss_cycle_y = self._lambda1 * loss_l1(y_x_y, y)

        chainer.report({'loss_rec': loss_cycle_y}, self.gen_g)
        chainer.report({'loss_rec': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_adv': loss_gen_g_adv}, self.gen_g)
        chainer.report({'loss_adv': loss_gen_f_adv}, self.gen_f)

        loss_gen = loss_gen_g_adv + loss_gen_f_adv + loss_cycle_x + loss_cycle_y
        loss_gen.backward()

        opt_f.update()
        opt_g.update()

        if self._iter >= self._learning_rate_anneal_trigger and \
                self._learning_rate_anneal > 0 and \
                self._iter % self._learning_rate_anneal_interval == 0:
            if opt_g.alpha > self._learning_rate_anneal:
                opt_g.alpha -= self._learning_rate_anneal
            if opt_f.alpha > self._learning_rate_anneal:
                opt_f.alpha -= self._learning_rate_anneal
            if opt_x.alpha > self._learning_rate_anneal:
                opt_x.alpha -= self._learning_rate_anneal
            if opt_y.alpha > self._learning_rate_anneal:
                opt_y.alpha -= self._learning_rate_anneal

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)

        self._buffer_x.serialize(serializer['buffer_x'])
        self._buffer_y.serialize(serializer['buffer_y'])

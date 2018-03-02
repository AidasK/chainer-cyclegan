import chainer
import six
from chainer import cuda, optimizers, serializers, Variable
import numpy as np
import chainer.functions as F

class HistoricalBuffer():
    def __init__(self, buffer_size=50, image_size=256, image_channels=3, gpu = -1):
        self._buffer_size = buffer_size
        self._img_size = image_size
        self._img_ch = image_channels
        self._cnt = 0
        self.gpu = gpu
        import numpy
        import cupy
        xp = numpy if gpu < 0 else cupy
        self._buffer = xp.zeros((self._buffer_size, self._img_ch, self._img_size, self._img_size)).astype("f")

    def get_and_update(self, data, prob=0.5):
        if self._buffer_size == 0:
            return data
        xp = chainer.cuda.get_array_module(data)

        if len(data) == 1:
            if self._cnt < self._buffer_size:
                self._buffer[self._cnt,:] = chainer.cuda.to_cpu(data[0,:]) if self.gpu == -1 else data[0,:]
                self._cnt += 1
                return data
            else:
                if np.random.rand() > prob:
                    self._buffer[np.random.randint(self._cnt), :] = chainer.cuda.to_cpu(data[0,:]) if self.gpu == -1 else data[0,:]
                    return data
                else:
                    return xp.expand_dims(xp.asarray(self._buffer[np.random.randint(self._cnt),:]),axis=0)
        else:
            data = xp.copy(data)
            use_buf = len(data) // 2
            indices_rand = np.random.permutation(len(data))

            avail_buf = min(self._cnt, use_buf)
            if avail_buf > 0:
                indices_use_buf = np.random.choice(self._cnt,avail_buf,replace=False)
                data[indices_rand[-avail_buf:],:] = xp.asarray(self._buffer[indices_use_buf,:])
            room_buf = self._buffer_size - self._cnt
            n_replace_buf = min(self._cnt,len(data)-avail_buf-room_buf)
            if n_replace_buf > 0:
                indices_replace_buf = np.random.choice(self._cnt,n_replace_buf,replace=False)
                self._buffer[indices_replace_buf,:] =  chainer.cuda.to_cpu(data[indices_rand[-avail_buf-n_replace_buf:-avail_buf],:]) \
                    if self.gpu == -1 else data[indices_rand[-avail_buf-n_replace_buf:-avail_buf],:]
            if room_buf > 0:
                n_fill_buf = min(room_buf, len(data)-avail_buf)
                self._buffer[self._cnt:self._cnt+n_fill_buf,:] = chainer.cuda.to_cpu(data[indices_rand[0:n_fill_buf],:]) \
                    if self.gpu == -1 else data[indices_rand[0:n_fill_buf],:]
                self._cnt += n_fill_buf
            return data

    def serialize(self, serializer):
        self._cnt = serializer('cnt', self._cnt)
        self.gpu = serializer('gpu', self.gpu)
        self._buffer = serializer('buffer', self._buffer)

class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        self._lambda1 = params['lambda1']
        self._lambda2 = params['lambda2']
        self._lambda_idt = params['lambda_idt']
        self._learning_rate_anneal = params['learning_rate_anneal']
        self._learning_rate_anneal_interval = params['learning_rate_anneal_interval']
        self._learning_rate_anneal_trigger = params['learning_rate_anneal_trigger']
        self._image_size = params['image_size']
        self._max_buffer_size = params['buffer_size']
        self.xp = self.gen_g.xp
        # self._cfmap_loss = params['cfmap_loss']
        # self._lambda_cfmap = params['lambda_cfmap']

        self._buffer_x = HistoricalBuffer(self._max_buffer_size, self._image_size)
        self._buffer_y = HistoricalBuffer(self._max_buffer_size, self._image_size)
        super(Updater, self).__init__(*args, **kwargs)

    def loss_func_rec_l1(self, x_out, t):
        return F.mean_absolute_error(x_out, t)

    def loss_func_adv_dis_fake(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 0.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def loss_func_adv_dis_real(self, y_real):
        target = Variable(
            self.xp.full(y_real.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_real, target)

    def loss_func_adv_gen(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def update_core(self):
        xp = self.gen_g.xp
        batch = self.get_iterator('main').next()

        batchsize = len(batch)

        x = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")
        y = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")

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
        y_x_copy = self._buffer_y.get_and_update(y_x.data)
        y_x_copy = Variable(y_x_copy)
        y_x_y = self.gen_g(y_x)

        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')

        #transformer optimization
        self.gen_f.cleargrads()
        self.gen_g.cleargrads()

        loss_gen_g_adv = self._lambda2 * self.loss_func_adv_gen(self.dis_y(x_y))
        loss_gen_f_adv = self._lambda2 * self.loss_func_adv_gen(self.dis_x(y_x))
        loss_cycle_x = self._lambda1 * self.loss_func_rec_l1(x_y_x, x)
        loss_cycle_y = self._lambda1 * self.loss_func_rec_l1(y_x_y, y)

        # if self._cfmap_loss in [0,2]:
        #     loss_cfmap_X = F.mean_squared_error(self.dis_y(x), self.dis_x(x_y)) * self._lambda_cfmap
        #     loss_cfmap_Y = F.mean_squared_error(self.dis_x(y), self.dis_y(y_x)) * self._lambda_cfmap
        #     loss_cfmap_gen = loss_cfmap_X + loss_cfmap_X
        # else:
        #     loss_cfmap_gen = 0
        # if self._cfmap_loss in [1, 2]:
        #     loss_cfmap_X = F.mean_squared_error(self.dis_y(x.data), self.dis_x(x_y.data)) * self._lambda_cfmap
        #     loss_cfmap_Y = F.mean_squared_error(self.dis_x(y.data), self.dis_y(y_x.data)) * self._lambda_cfmap
        #     loss_cfmap_dis = (loss_cfmap_X + loss_cfmap_X) * 0.5
        # else:
        #     loss_cfmap_dis = 0

        if self._lambda_idt > 0:
            idtY = self.gen_g(y)
            loss_idtY = F.mean_absolute_error(idtY,y)
            idtX = self.gen_f(x)
            loss_idtX = F.mean_absolute_error(idtX,x)
            loss_idt = (loss_idtX + loss_idtY) * self._lambda1 * self._lambda_idt
        else:
            loss_idtY = 0
            loss_idtX = 0
            loss_idt = 0

        chainer.report({'loss_rec': loss_cycle_y}, self.gen_g)
        chainer.report({'loss_rec': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_adv': loss_gen_g_adv}, self.gen_g)
        chainer.report({'loss_adv': loss_gen_f_adv}, self.gen_f)
        chainer.report({'loss_idt': loss_idtY}, self.gen_g)
        chainer.report({'loss_idt': loss_idtX}, self.gen_f)
        # if self._cfmap_loss != None:
        #     chainer.report({'loss_cfmap_X': loss_cfmap_X})
        #     chainer.report({'loss_cfmap_Y': loss_cfmap_Y})

        loss_gen = loss_gen_g_adv + loss_gen_f_adv + loss_cycle_x + loss_cycle_y + loss_idt #+ loss_cfmap_gen
        loss_gen.backward()

        opt_f.update()
        opt_g.update()


        # dicriminator optimization
        self.dis_y.cleargrads()
        self.dis_x.cleargrads()

        loss_dis_y_fake = self.loss_func_adv_dis_fake(self.dis_y(x_y_copy))
        loss_dis_y_real = self.loss_func_adv_dis_real(self.dis_y(y))
        loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5
        chainer.report({'loss': loss_dis_y}, self.dis_y)

        loss_dis_x_fake = self.loss_func_adv_dis_fake(self.dis_x(y_x_copy))
        loss_dis_x_real = self.loss_func_adv_dis_real(self.dis_x(x))
        loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5
        chainer.report({'loss': loss_dis_x}, self.dis_x)

        loss_dis_y.backward()
        loss_dis_x.backward()

        # if isinstance(loss_cfmap_dis, Variable):
        #     loss_cfmap_dis.backward()

        opt_y.update()
        opt_x.update()

        if self.iteration >= self._learning_rate_anneal_trigger and \
                self._learning_rate_anneal > 0 and \
                self.iteration % self._learning_rate_anneal_interval == 0:
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

class Updater_SimGAN(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        self._lambda1 = params['lambda1']
        self._lambda2 = params['lambda2']
        self._lambda_idt = params['lambda_idt']
        self._learning_rate_anneal = params['learning_rate_anneal']
        self._learning_rate_anneal_interval = params['learning_rate_anneal_interval']
        self._learning_rate_anneal_trigger = params['learning_rate_anneal_trigger']
        self._image_size = params['image_size']
        self._max_buffer_size = params['buffer_size']
        self.xp = self.gen_g.xp
        # self._cfmap_loss = params['cfmap_loss']
        # self._lambda_cfmap = params['lambda_cfmap']

        self._buffer_x = HistoricalBuffer(self._max_buffer_size, self._image_size)
        self._buffer_y = HistoricalBuffer(self._max_buffer_size, self._image_size)
        super(Updater_SimGAN, self).__init__(*args, **kwargs)

    def loss_func_rec_l1(self, x_out, t):
        return F.mean_absolute_error(x_out, t)

    def loss_func_adv_dis_fake(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 0.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def loss_func_adv_dis_real(self, y_real):
        target = Variable(
            self.xp.full(y_real.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_real, target)

    def loss_func_adv_gen(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def update_core(self):
        xp = self.gen_g.xp
        batch = self.get_iterator('main').next()

        batchsize = len(batch)

        x = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")
        y = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")

        for i in range(batchsize):
            x[i, :] = xp.asarray(batch[i][0])
            y[i, :] = xp.asarray(batch[i][1])

        x = Variable(x)
        y = Variable(y)

        x_y = self.gen_g(x)
        x_y_copy = self._buffer_x.get_and_update(x_y.data)
        x_y_copy = Variable(x_y_copy)
        # x_y_x = self.gen_f(x_y)

        # y_x = self.gen_f(y)
        # y_x_copy = self._buffer_y.get_and_update(y_x.data)
        # y_x_copy = Variable(y_x_copy)
        # y_x_y = self.gen_g(y_x)

        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')

        #transformer optimization
        # self.gen_f.cleargrads()
        self.gen_g.cleargrads()

        loss_gen_g_adv = self._lambda2 * self.loss_func_adv_gen(self.dis_y(x_y))
        loss_gen_g_l1 = self._lambda1 * self._lambda_idt * F.mean_absolute_error(x,x_y)
        # loss_gen_f_adv = self._lambda2 * self.loss_func_adv_gen(self.dis_x(y_x))
        # loss_cycle_x = self._lambda1 * self.loss_func_rec_l1(x_y_x, x)
        # loss_cycle_y = self._lambda1 * self.loss_func_rec_l1(y_x_y, y)

        # if self._cfmap_loss in [0,2]:
        #     loss_cfmap_X = F.mean_squared_error(self.dis_y(x), self.dis_x(x_y)) * self._lambda_cfmap
        #     loss_cfmap_Y = F.mean_squared_error(self.dis_x(y), self.dis_y(y_x)) * self._lambda_cfmap
        #     loss_cfmap_gen = loss_cfmap_X + loss_cfmap_X
        # else:
        #     loss_cfmap_gen = 0
        # if self._cfmap_loss in [1, 2]:
        #     loss_cfmap_X = F.mean_squared_error(self.dis_y(x.data), self.dis_x(x_y.data)) * self._lambda_cfmap
        #     loss_cfmap_Y = F.mean_squared_error(self.dis_x(y.data), self.dis_y(y_x.data)) * self._lambda_cfmap
        #     loss_cfmap_dis = (loss_cfmap_X + loss_cfmap_X) * 0.5
        # else:
        #     loss_cfmap_dis = 0

        if self._lambda_idt > 0:
            idtY = self.gen_g(y)
            loss_idtY = F.mean_absolute_error(idtY,y)
            # idtX = self.gen_f(x)
            # loss_idtX = F.mean_absolute_error(idtX,x)
            loss_idt = loss_idtY * self._lambda1 * self._lambda_idt
        else:
            loss_idtY = 0
            loss_idtX = 0
            loss_idt = 0

        # chainer.report({'loss_rec': loss_cycle_y}, self.gen_g)
        # chainer.report({'loss_rec': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_adv': loss_gen_g_adv}, self.gen_g)
        # chainer.report({'loss_adv': loss_gen_f_adv}, self.gen_f)
        chainer.report({'loss_idt': loss_idtY}, self.gen_g)
        chainer.report({'loss_sim_l1': loss_gen_g_l1}, self.gen_g)
        # chainer.report({'loss_idt': loss_idtX}, self.gen_f)
        # if self._cfmap_loss != None:
        #     chainer.report({'loss_cfmap_X': loss_cfmap_X})
        #     chainer.report({'loss_cfmap_Y': loss_cfmap_Y})

        loss_gen = loss_gen_g_adv  + loss_idt +loss_gen_g_l1#+ loss_cfmap_gen
        loss_gen.backward()

        # opt_f.update()
        opt_g.update()


        # dicriminator optimization
        self.dis_y.cleargrads()
        # self.dis_x.cleargrads()

        loss_dis_y_fake = self.loss_func_adv_dis_fake(self.dis_y(x_y_copy))
        loss_dis_y_real = self.loss_func_adv_dis_real(self.dis_y(y))
        loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5
        chainer.report({'loss': loss_dis_y}, self.dis_y)

        # loss_dis_x_fake = self.loss_func_adv_dis_fake(self.dis_x(y_x_copy))
        # loss_dis_x_real = self.loss_func_adv_dis_real(self.dis_x(x))
        # loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5
        # chainer.report({'loss': loss_dis_x}, self.dis_x)

        loss_dis_y.backward()
        # loss_dis_x.backward()

        # if isinstance(loss_cfmap_dis, Variable):
        #     loss_cfmap_dis.backward()

        opt_y.update()
        # opt_x.update()

        if self.iteration >= self._learning_rate_anneal_trigger and \
                self._learning_rate_anneal > 0 and \
                self.iteration % self._learning_rate_anneal_interval == 0:
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

class Updater_gt_l1(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen_g, self.gen_f, self.dis_x, self.dis_y = kwargs.pop('models')
        params = kwargs.pop('params')
        self._lambda1 = params['lambda1']
        self._lambda2 = params['lambda2']
        self._lambda_idt = params['lambda_idt']
        self._learning_rate_anneal = params['learning_rate_anneal']
        self._learning_rate_anneal_interval = params['learning_rate_anneal_interval']
        self._learning_rate_anneal_trigger = params['learning_rate_anneal_trigger']
        self._image_size = params['image_size']
        self._max_buffer_size = params['buffer_size']
        self.xp = self.gen_g.xp
        # self._cfmap_loss = params['cfmap_loss']
        # self._lambda_cfmap = params['lambda_cfmap']

        self._buffer_x = HistoricalBuffer(self._max_buffer_size, self._image_size)
        self._buffer_y = HistoricalBuffer(self._max_buffer_size, self._image_size)
        super(Updater_gt_l1, self).__init__(*args, **kwargs)

    def loss_func_rec_l1(self, x_out, t):
        return F.mean_absolute_error(x_out, t)

    def loss_func_adv_dis_fake(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 0.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def loss_func_adv_dis_real(self, y_real):
        target = Variable(
            self.xp.full(y_real.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_real, target)

    def loss_func_adv_gen(self, y_fake):
        target = Variable(
            self.xp.full(y_fake.data.shape, 1.0).astype('f'))
        return F.mean_squared_error(y_fake, target)

    def update_core(self):
        xp = self.gen_g.xp
        batch = self.get_iterator('main').next()

        batchsize = len(batch)

        # x = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")
        # y = xp.zeros((batchsize, 3, self._image_size, self._image_size)).astype("f")
        #
        # for i in range(batchsize):
        #     x[i, :] = xp.asarray(batch[i][0])
        #     y[i, :] = xp.asarray(batch[i][1])

        x_images = [batch[i][0] for i in range(batchsize)]
        x_gt_maps = [batch[i][1] for i in range(batchsize)]
        y_images = [batch[i][2] for i in range(batchsize)]

        x = xp.stack(x_images)
        x_gt_maps =xp.stack(x_gt_maps)
        y = xp.stack(y_images)

        x = Variable(x)
        x_gt_maps = Variable(x_gt_maps)
        y = Variable(y)

        x_y = self.gen_g(x)
        x_y_copy = self._buffer_x.get_and_update(x_y.data)
        x_y_copy = Variable(x_y_copy)
        x_y_x = self.gen_f(x_y)

        y_x = self.gen_f(y)
        y_x_copy = self._buffer_y.get_and_update(y_x.data)
        y_x_copy = Variable(y_x_copy)
        y_x_y = self.gen_g(y_x)

        opt_g = self.get_optimizer('gen_g')
        opt_f = self.get_optimizer('gen_f')
        opt_x = self.get_optimizer('dis_x')
        opt_y = self.get_optimizer('dis_y')

        #transformer optimization
        self.gen_f.cleargrads()
        self.gen_g.cleargrads()

        loss_gen_g_adv = self._lambda2 * self.loss_func_adv_gen(self.dis_y(x_y))
        loss_gen_f_adv = self._lambda2 * self.loss_func_adv_gen(self.dis_x(y_x))
        loss_cycle_x = self._lambda1 * self.loss_func_rec_l1(x_y_x, x)
        loss_cycle_y = self._lambda1 * self.loss_func_rec_l1(y_x_y, y)

        loss_gen_g_gt_l1 = self._lambda1 * (F.absolute_error(x,x_y) * x_gt_maps / np.prod(x.data.shape))
        loss_gen_f_gt_l1 = self._lambda1 * (F.absolute_error(x_y,x_y_x) * x_gt_maps / np.prod(x.data.shape))
        loss_gen_gt_l1 = loss_gen_g_gt_l1 + loss_gen_f_gt_l1

        # if self._cfmap_loss in [0,2]:
        #     loss_cfmap_X = F.mean_squared_error(self.dis_y(x), self.dis_x(x_y)) * self._lambda_cfmap
        #     loss_cfmap_Y = F.mean_squared_error(self.dis_x(y), self.dis_y(y_x)) * self._lambda_cfmap
        #     loss_cfmap_gen = loss_cfmap_X + loss_cfmap_X
        # else:
        #     loss_cfmap_gen = 0
        # if self._cfmap_loss in [1, 2]:
        #     loss_cfmap_X = F.mean_squared_error(self.dis_y(x.data), self.dis_x(x_y.data)) * self._lambda_cfmap
        #     loss_cfmap_Y = F.mean_squared_error(self.dis_x(y.data), self.dis_y(y_x.data)) * self._lambda_cfmap
        #     loss_cfmap_dis = (loss_cfmap_X + loss_cfmap_X) * 0.5
        # else:
        #     loss_cfmap_dis = 0

        if self._lambda_idt > 0:
            idtY = self.gen_g(y)
            loss_idtY = F.mean_absolute_error(idtY,y)
            idtX = self.gen_f(x)
            loss_idtX = F.mean_absolute_error(idtX,x)
            loss_idt = (loss_idtX + loss_idtY) * self._lambda1 * self._lambda_idt
        else:
            loss_idtY = 0
            loss_idtX = 0
            loss_idt = 0

        chainer.report({'loss_rec': loss_cycle_y}, self.gen_g)
        chainer.report({'loss_rec': loss_cycle_x}, self.gen_f)
        chainer.report({'loss_adv': loss_gen_g_adv}, self.gen_g)
        chainer.report({'loss_adv': loss_gen_f_adv}, self.gen_f)
        chainer.report({'loss_idt': loss_idtY}, self.gen_g)
        chainer.report({'loss_idt': loss_idtX}, self.gen_f)
        chainer.report({'loss_gen_gt_l1': loss_gen_g_gt_l1}, self.gen_g)
        chainer.report({'loss_gen_gt_l1': loss_gen_f_gt_l1}, self.gen_f)
        # if self._cfmap_loss != None:
        #     chainer.report({'loss_cfmap_X': loss_cfmap_X})
        #     chainer.report({'loss_cfmap_Y': loss_cfmap_Y})

        loss_gen = loss_gen_g_adv + loss_gen_f_adv + loss_cycle_x + loss_cycle_y + loss_idt + loss_gen_gt_l1#+ loss_cfmap_gen
        loss_gen.backward()

        opt_f.update()
        opt_g.update()


        # dicriminator optimization
        self.dis_y.cleargrads()
        self.dis_x.cleargrads()

        loss_dis_y_fake = self.loss_func_adv_dis_fake(self.dis_y(x_y_copy))
        loss_dis_y_real = self.loss_func_adv_dis_real(self.dis_y(y))
        loss_dis_y = (loss_dis_y_fake + loss_dis_y_real) * 0.5
        chainer.report({'loss': loss_dis_y}, self.dis_y)

        loss_dis_x_fake = self.loss_func_adv_dis_fake(self.dis_x(y_x_copy))
        loss_dis_x_real = self.loss_func_adv_dis_real(self.dis_x(x))
        loss_dis_x = (loss_dis_x_fake + loss_dis_x_real) * 0.5
        chainer.report({'loss': loss_dis_x}, self.dis_x)

        loss_dis_y.backward()
        loss_dis_x.backward()

        # if isinstance(loss_cfmap_dis, Variable):
        #     loss_cfmap_dis.backward()

        opt_y.update()
        opt_x.update()

        if self.iteration >= self._learning_rate_anneal_trigger and \
                self._learning_rate_anneal > 0 and \
                self.iteration % self._learning_rate_anneal_interval == 0:
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

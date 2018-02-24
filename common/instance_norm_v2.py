import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import functions
from chainer import initializers
from chainer import link
from chainer.functions.normalization import batch_normalization
from chainer.utils import argument
from chainer import variable

class Parameter_scale(variable.Parameter):
    def __init__(self, initializer=None, shape=None, name=None,norm_grad = False):
        super(Parameter_scale, self).__init__(initializer, shape, name)
        self.norm_grad = norm_grad
        self._n_batch = 0

    def update(self):
        """Updates the data array using the gradient and the update rule.
        This method updates the parameter using the attached update rule.
        """
        if self.norm_grad:
            if self._n_batch != 0:
                if self.grad is not None:
                    self.grad = self.grad / self._n_batch
                    self._n_batch = 0
                    # print("debug: normalized grad")
                    # print(self.grad)
                else:
                    print("Warning in {0}: Grad has not been calculated yet. n_batch is not initialized.".format(self.__class__.__name__))
            else:
                print("Warning in {0}: update() might have been called multiple times in one iteration. \
                Grad is not normalized.".format(self.__class__.__name__))

        if self.update_rule is not None:
            self.update_rule.update(self)

    @property
    def n_batch(self):
        return self._n_batch

    # @n_batch.setter
    # def n_batch(self,n):
    #     self._n_batch += n

    def add_batch(self,n):
        self._n_batch += n

class InstanceNormalization(link.Link):
    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None, norm_grad = False):
        super(InstanceNormalization, self).__init__()
        self.avg_mean = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_mean')
        self.avg_var = numpy.zeros(size, dtype=dtype)
        self.register_persistent('avg_var')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.norm_grad = norm_grad

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = dtype
                self.gamma = Parameter_scale(initial_gamma, size, norm_grad=self.norm_grad)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = Parameter_scale(initial_beta, size, norm_grad=self.norm_grad)

    def __call__(self, x, **kwargs):
        """__call__(self, x, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluation during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', False)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        # check argument
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
                         'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        # reshape input x
        original_shape = x.shape
        # batch_size, n_ch = original_shape[:2]
        batch_size = original_shape[0]
        # new_shape = (1, batch_size * n_ch) + original_shape[2:]
        # reshaped_x = functions.reshape(x, new_shape)
        reshaped_x = functions.expand_dims(x, axis=0)

        if hasattr(self, 'gamma'):
            gamma = self.gamma
            if self.norm_grad:
                gamma.add_batch(batch_size)
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
            if self.norm_grad:
                beta.add_batch(batch_size)
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))

        # mean = chainer.as_variable(self.xp.hstack([self.avg_mean] * batch_size))
        mean = self.xp.stack((self.avg_mean,) * batch_size)
        # var = chainer.as_variable(self.xp.hstack([self.avg_var] * batch_size))
        var = self.xp.stack((self.avg_var,) * batch_size)
        # gamma = chainer.as_variable(self.xp.hstack([gamma.array] * batch_size))
        gamma = functions.stack((gamma,) * batch_size)
        # beta = chainer.as_variable(self.xp.hstack([beta.array] * batch_size))
        beta = functions.stack((beta,) * batch_size)

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = batch_normalization.BatchNormalizationFunction(
                self.eps, mean, var, decay)
            ret = func(reshaped_x, gamma, beta)

            self.avg_mean[...] = func.running_mean.mean(axis=0)
            self.avg_var[...] = func.running_var.mean(axis=0)

            # ret = functions.batch_normalization(
            #     reshaped_x, gamma, beta, eps=self.eps, running_mean=mean,
            #     running_var=var, decay=decay)
            #
            # # store running mean and var
            # self.avg_mean[...] = mean.mean(axis=0)
            # self.avg_var[...] = var.mean(axis=0)

        else:
            # Use running average statistics or fine-tuned statistics.
            # mean = variable.Variable(mean)
            # var = variable.Variable(var)
            head_ndim = gamma.ndim + 1
            axis = (0,) + tuple(range(head_ndim, reshaped_x.ndim))
            mean = reshaped_x.data.mean(axis=axis)
            var = reshaped_x.data.var(axis=axis)
            # var += self.eps
            ret = functions.fixed_batch_normalization(
                reshaped_x, gamma, beta, mean, var, self.eps)

        # ret is normalized input x
        return functions.reshape(ret, original_shape)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)
    base_shape = [1, 3]
    with chainer.using_config('debug', True):
        for i, n_element in enumerate([32, 32, 32]):
            base_shape.append(n_element)
            print('# {} th: input shape: {}'.format(i, base_shape))
            x_array = np.random.normal(size=base_shape).astype(np.float32)
            x = chainer.Variable(x_array)
            layer = InstanceNormalization(base_shape[1],norm_grad=True)
            y = layer(x)
            # calculate y_hat manually
            axes = tuple(range(2, len(base_shape)))
            x_mean = np.mean(x_array, axis=axes, keepdims=True)
            x_var = np.var(x_array, axis=axes, keepdims=True) + 1e-5
            x_std = np.sqrt(x_var)
            y_hat = (x_array - x_mean) / x_std
            diff = y.data - y_hat
            print('*** diff ***')
            print('\tmean: {:03f},\n\tstd: {:.03f}'.format(
                np.mean(diff), np.std(diff)))

        y_ = chainer.links.BatchNormalization(base_shape[1])(x)

        print((y.data == y_.data).any())

        print(type(y))
        loss = functions.sum(y)
        layer.cleargrads()
        loss.backward()
        print(layer.gamma.grad)
        print(layer.beta.grad)
        print(layer.avg_mean)
        print(layer.avg_var)
        opt = chainer.optimizers.SGD()
        opt.setup(layer)
        opt.update()

        base_shape = [10, 3]
        with chainer.using_config('train', False):
            print('\n# test mode\n')
            for i, n_element in enumerate([32, 32, 32]):
                base_shape.append(n_element)
                print('# {} th: input shape: {}'.format(i, base_shape))
                x_array = np.random.normal(size=base_shape).astype(np.float32)
                x = chainer.Variable(x_array)
                layer = InstanceNormalization(base_shape[1])
                y = layer(x)
                axes = tuple(range(2, len(base_shape)))
                x_mean = np.mean(x_array, axis=axes, keepdims=True)
                x_var = np.var(x_array, axis=axes, keepdims=True) + 1e-5
                x_std = np.sqrt(x_var)
                y_hat = (x_array - x_mean) / x_std
                diff = y.data - y_hat
                print('*** diff ***')
                print('\tmean: {:03f},\n\tstd: {:.03f}'.format(
                    np.mean(diff), np.std(diff)))

"""
○ → python instance_norm.py
# 0 th: input shape: [10, 3, 32]
*** diff ***
        mean: -0.000000,
        std: 0.000
# 1 th: input shape: [10, 3, 32, 32]
*** diff ***
        mean: -0.000000,
        std: 0.000
# 2 th: input shape: [10, 3, 32, 32, 32]
*** diff ***
        mean: -0.000000,
        std: 0.000

# test mode
# 0 th: input shape: [10, 3, 32]
*** diff ***
        mean: 14.126040,
        std: 227.823
# 1 th: input shape: [10, 3, 32, 32]
*** diff ***
        mean: -0.286635,
        std: 221.926
# 2 th: input shape: [10, 3, 32, 32, 32]
*** diff ***
        mean: -0.064297,
        std: 222.492
"""

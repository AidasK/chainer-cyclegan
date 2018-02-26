import numpy

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

        original_shape = x.shape
        batch_size = original_shape[0]
        # reshape input x if batchsize > 1
        if batch_size > 1:
            reshaped_x = functions.expand_dims(x, axis=0)
        else:
            reshaped_x = x

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

        #align shapes if x was reshaped
        if batch_size > 1:
            mean = self.xp.stack((self.avg_mean,) * batch_size)
            var = self.xp.stack((self.avg_var,) * batch_size)
            gamma = functions.stack((gamma,) * batch_size)
            beta = functions.stack((beta,) * batch_size)
        else:
            mean = self.xp.asarray(self.avg_mean)
            var = self.xp.asarray(self.avg_var)

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = batch_normalization.BatchNormalizationFunction(
                self.eps, mean, var, decay)
            ret = func(reshaped_x, gamma, beta)

        else:
            head_ndim = gamma.ndim + 1
            axis = (0,) + tuple(range(head_ndim, reshaped_x.ndim))
            mean = reshaped_x.data.mean(axis=axis)
            var = reshaped_x.data.var(axis=axis)
            ret = functions.fixed_batch_normalization(
                reshaped_x, gamma, beta, mean, var, self.eps)

        # ret is normalized input x
        if batch_size > 1:
            ret = functions.reshape(ret, original_shape)
        return ret

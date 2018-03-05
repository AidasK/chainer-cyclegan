import chainer
from chainer import Variable, cuda
import numpy as np
import os
from common.utils.save_images import save_images_grid

def _sample_generation(gen_g, gen_f, dataset, eval_folder, batch_size, random, indices, trigger, count):
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)

    xp = gen_f.xp

    if random:
        datasize_A = dataset.len_A
        datasize_B = dataset.len_B
        indices_A = np.random.choice(datasize_A, batch_size, replace=False)
        indices_B = np.random.choice(datasize_B, batch_size, replace=False)
    else:
        indices_A = indices[0]
        indices_B = indices[1]
    batches_A = [xp.asarray(dataset.get_example_raw_A(i)) for i in indices_A]
    batches_B = [xp.asarray(dataset.get_example_raw_B(i)) for i in indices_B]

    x = xp.stack(batches_A)
    y = xp.stack(batches_B)

    x = Variable(x)
    y = Variable(y)

    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        x_y = gen_g(x)
        y_x = gen_f(y)
        x_y_x = gen_f(x_y)
        y_x_y = gen_g(y_x)

    imgs = xp.concatenate((x.data, x_y.data, x_y_x.data, y.data, y_x.data, y_x_y.data), axis=0)

    if trigger == 'iteration':
        prefix = 'iter_'
    elif trigger == 'epoch':
        prefix = 'epoch_'
    filename = prefix + str(count) + ".jpg"

    save_images_grid(imgs, path=os.path.join(eval_folder, filename))

def visualization(gen_g, gen_f, dataset, eval_folder=".", batch_size=1, random = True, indices = None):
    @chainer.training.make_extension()
    def samples_generation(trainer):
        trigger = 'iteration'
        count = getattr(trainer.updater, trigger)
        _sample_generation(gen_g, gen_f, dataset, eval_folder, batch_size, random, indices, trigger, count)

    return samples_generation

#!/usr/bin/env python
import argparse
import os
import sys
import chainer
from chainer import training
from chainer import cuda, serializers
from chainer.training import extensions
from updater import *
import common.datasets as datasets
from common.models.discriminators import *
from common.models.transformers import *
from common.evaluation.cyclegan import *
from common.utils import *

def main():
    parser = argparse.ArgumentParser(
        description='Train CycleGAN')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--max_iter', '-m', type=int, default=200000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_interval', type=int, default=400,
                        help='Interval of evaluating generator')

    parser.add_argument('--snapshot_interval', type=int, default=4000,
                        help='Interval of model snapshot')

    parser.add_argument("--learning_rate_g", type=float, default=0.0002,
                        help="Learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=0.0002,
                        help="Learning rate for discriminator")

    parser.add_argument("--data_train_x", default='', help='path of train data of x')
    parser.add_argument("--data_train_y", default='', help='path of train data of y')
    parser.add_argument("--data_test_x", type=str, help='path of test data of x')
    parser.add_argument("--data_test_y", type=str, help='path of test data of y')

    parser.add_argument("--resume", type = str, help='trainer snapshot to be resumed')

    parser.add_argument("--load_gen_f_model", default='', help='load generator model')
    parser.add_argument("--load_gen_g_model", default='', help='load generator model')
    parser.add_argument("--load_dis_x_model", default='', help='load discriminator model')
    parser.add_argument("--load_dis_y_model", default='', help='load discriminator model')

    parser.add_argument("--resize_to", type=int, default=280, help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=256, help='crop the resized image to')

    parser.add_argument("--lambda1", type=float, default=10.0, help='lambda for reconstruction loss')
    parser.add_argument("--lambda2", type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument("--lambda_idt", type=float, default=0.5, help='lambda for identity mapping loss')

    parser.add_argument("--learning_rate_anneal", type=float, default=0.000002, help='anneal the learning rate')
    parser.add_argument("--learning_rate_anneal_interval", type=int, default=1000, help='interval of learning rate anneal')
    parser.add_argument("--learning_rate_anneal_trigger", type=int, default=100000, help='trigger of learning rate anneal')

    parser.add_argument("--norm", type=str, default='instance', choices = ['instance','bn','None'], help='normalization method')
    parser.add_argument("--upsample", type=str, default='up_unpooling', choices=['up_unpooling', 'up_subpixel'],
                        help='upsample method')
    parser.add_argument("--reflect", type=int, default=0, choices = [0,1,2], help='flag of using reflect padding - 0: non use, 1: at the begining, 2: each time')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    gen_g = ResNetImageTransformer(norm_func=args.norm, upsampling=args.upsample, reflect=args.reflect)
    gen_f = ResNetImageTransformer(norm_func=args.norm, upsampling=args.upsample, reflect=args.reflect)
    dis_x = DCGANDiscriminator(base_size=64, conv_as_last=True)
    dis_y = DCGANDiscriminator(base_size=64, conv_as_last=True)

    if args.load_gen_g_model != '':
        serializers.load_npz(args.load_gen_g_model, gen_g)
        print("Generator G(X->Y) model loaded")

    if args.load_gen_f_model != '':
        serializers.load_npz(args.load_gen_f_model, gen_f)
        print("Generator F(Y->X) model loaded")

    if args.load_dis_x_model != '':
        serializers.load_npz(args.load_dis_x_model, dis_x)
        print("Discriminator X model loaded")

    if args.load_dis_y_model != '':
        serializers.load_npz(args.load_dis_y_model, dis_y)
        print("Discriminator Y model loaded")

    if args.gpu >= 0:
        gen_g.to_gpu()
        gen_f.to_gpu()
        dis_x.to_gpu()
        dis_y.to_gpu()
        print("use gpu {}".format(args.gpu))

    opt_g=make_adam(gen_g, lr=args.learning_rate_g, beta1=0.5)
    opt_f=make_adam(gen_f, lr=args.learning_rate_g, beta1=0.5)
    opt_x=make_adam(dis_x, lr=args.learning_rate_d, beta1=0.5)
    opt_y=make_adam(dis_y, lr=args.learning_rate_d, beta1=0.5)

    train_dataset = datasets.image_pairs_train(args.data_train_x, args.data_train_y,
            resize_to=args.resize_to, crop_to=args.crop_to)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, args.batch_size, n_processes=4)

    if args.data_test_x:
        test_dataset = datasets.image_pairs_train(args.data_test_x, args.data_test_y,
            resize_to=args.crop_to, crop_to=args.crop_to)
        test_iter = chainer.iterators.SerialIterator(test_dataset, 1)
    else:
        test_iter = chainer.iterators.SerialIterator(train_dataset, 1)

    # Set up a trainer
    updater = Updater(
        models=(gen_g, gen_f, dis_x, dis_y),
        iterator={
            'main': train_iter,
            'test': test_iter
            },
        optimizer={
            'gen_g': opt_g,
            'gen_f': opt_f,
            'dis_x': opt_x,
            'dis_y': opt_y
            },
        device=args.gpu,
        params={
            'lambda1': args.lambda1,
            'lambda2': args.lambda2,
            'lambda_idt': args.lambda_idt,
            'image_size' : args.crop_to,
            'buffer_size' : 50,
            'learning_rate_anneal' : args.learning_rate_anneal,
            'learning_rate_anneal_trigger' : args.learning_rate_anneal_trigger,
            'learning_rate_anneal_interval' : args.learning_rate_anneal_interval,
        })

    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    model_save_interval = (args.snapshot_interval, 'iteration')
    trainer.extend(extensions.snapshot_object(
        gen_g, 'gen_g{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        gen_f, 'gen_f{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_x, 'dis_x{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_y, 'dis_y{.updater.iteration}.npz'), trigger=model_save_interval)

    trainer.extend(extensions.snapshot(), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'gen_g/loss_rec', 'gen_f/loss_rec', 'gen_g/loss_adv',
                'gen_f/loss_adv', 'gen_g/loss_idt', 'gen_f/loss_idt', 'dis_x/loss', 'dis_y/loss']

    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(20, 'iteration')))
    trainer.extend(extensions.PrintReport(log_keys), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=50))

    eval_interval = (args.eval_interval, 'iteration')
    trainer.extend(
        cyclegan_sampling(gen_g, gen_f, test_iter, args.out+"/preview/", 1),
        trigger=eval_interval
    )

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()

if __name__ == '__main__':
    main()

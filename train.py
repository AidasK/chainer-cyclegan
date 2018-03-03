#!/usr/bin/env python
import argparse

from chainer import training
from chainer.training import extensions

import common.datasets as datasets
from common.evaluation.visualization import *
from common.models.discriminators import *
from common.models.transformers import *
from common.utils import *
from updater import *

import matplotlib
matplotlib.use('Agg')

def main():
    parser = argparse.ArgumentParser(
        description='Train CycleGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=1)
    parser.add_argument('--max_iter', '-m', type=int, default=200000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_interval', type=int, default=400,
                        help='Interval of evaluating generator')
    parser.add_argument('--vis_batch', type=int, default=1,
                        help='number of images for visualization')

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

    parser.add_argument("--load_gen_f_model", type=str, help='load generator model')
    parser.add_argument("--load_gen_g_model", type=str, help='load generator model')
    parser.add_argument("--load_dis_x_model", type=str, help='load discriminator model')
    parser.add_argument("--load_dis_y_model", type=str, help='load discriminator model')

    parser.add_argument("--resize_to", type=int, default=286, help='resize the image to')
    parser.add_argument("--crop_to", type=int, default=256, help='crop the resized image to')

    parser.add_argument("--lambda1", type=float, default=10.0, help='lambda for reconstruction loss')
    parser.add_argument("--lambda2", type=float, default=1.0, help='lambda for adversarial loss')
    parser.add_argument("--lambda_idt", type=float, default=0.5, help='lambda for identity mapping loss')

    parser.add_argument("--bufsize", type=int, default=50, help='size of buffer')

    # parser.add_argument("--cfmap_loss", type=int, choices = [0,1,2], help='use of cfmap loss 0: penalize gen, 1: penalize dis, 2: penalize both')
    # parser.add_argument("--lambda_cfmap", type=float, default=1.0,
    #                     help='lambda for cfmap loss')

    parser.add_argument("--learning_rate_anneal", type=float, default=0.000002, help='anneal the learning rate')
    parser.add_argument("--learning_rate_anneal_interval", type=int, default=1000, help='interval of learning rate anneal')
    parser.add_argument("--learning_rate_anneal_trigger", type=int, default=100000, help='trigger of learning rate anneal')

    parser.add_argument("--norm", type=str, default='instance', choices = ['instance','bn','None'], help='normalization method')
    parser.add_argument("--reflect", type=int, choices = [0,1,2],default=2, help='reflect padding setting 0: no use, 1: at the beginning, 2: each time')
    # parser.add_argument("--norm_noaffine", action='store_true')
    # parser.add_argument("--norm_gnorm", action='store_true')
    parser.add_argument("--method", type=str, default='default', choices=['default', 'SimGAN', 'GT_L1','SimGAN_GT_L1'],
                        help='updater method')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    if args.norm == 'None': args.norm = None

    gen_g = ResNetImageTransformer(norm=args.norm, reflect=args.reflect)
    gen_f = ResNetImageTransformer(norm=args.norm, reflect=args.reflect)
    dis_x = DCGANDiscriminator(norm=args.norm)
    dis_y = DCGANDiscriminator(norm=args.norm)

    if args.load_gen_g_model:
        serializers.load_npz(args.load_gen_g_model, gen_g)
        print("Generator G(X->Y) model loaded")

    if args.load_gen_f_model:
        serializers.load_npz(args.load_gen_f_model, gen_f)
        print("Generator F(Y->X) model loaded")

    if args.load_dis_x_model:
        serializers.load_npz(args.load_dis_x_model, dis_x)
        print("Discriminator X model loaded")

    if args.load_dis_y_model:
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

    if args.method == "GT_L1":
        dataset_class = datasets.source_target_dataset
    else:
        dataset_class = datasets.image_pairs_train

    train_dataset = dataset_class(args.data_train_x, args.data_train_y,
            resize_to=args.resize_to, crop_to=args.crop_to)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, args.batchsize, n_processes=4)

    if args.data_test_x:
        test_dataset = datasets.image_pairs_train(args.data_test_x, args.data_test_y,
            resize_to=args.crop_to, crop_to=args.crop_to)

    # Set up a trainer
    if args.method == "SimGAN":
        updater_choice = Updater_SimGAN
    elif args.method == "GT_L1":
        updater_choice = Updater_gt_l1
    elif args.method == 'SimGAN_GT_L1':
        updater_choice = Updater_SimGAN_gt_l1
    else:
        updater_choice = Updater

    updater = updater_choice(
        models=(gen_g, gen_f, dis_x, dis_y),
        iterator={
            'main': train_iter,
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
            'buffer_size' : args.bufsize,
            'learning_rate_anneal' : args.learning_rate_anneal,
            'learning_rate_anneal_trigger' : args.learning_rate_anneal_trigger,
            'learning_rate_anneal_interval' : args.learning_rate_anneal_interval,
            # 'cfmap_loss' : args.cfmap_loss,
            # 'lambda_cfmap' : args.lambda_cfmap,
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
    # if args.cfmap_loss != None:
    #     log_keys += ['loss_cfmap_X', 'loss_cfmap_Y']

    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(20, 'iteration')))
    trainer.extend(extensions.PrintReport(log_keys), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=50))

    plotreport_keys = ['epoch', 'gen_g/loss_rec', 'gen_f/loss_rec', 'gen_g/loss_adv',
                 'gen_f/loss_adv', 'gen_g/loss_idt', 'gen_f/loss_idt', 'dis_x/loss', 'dis_y/loss']

    if args.method in ["SimGAN",'SimGAN_GT_L1']:
        plotreport_keys += ['gen_g/loss_sim_l1']
    if args.method == "GT_L1":
        plotreport_keys += ['gen_g/loss_gen_gt_l1', 'gen_f/loss_gen_gt_l1']

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                plotreport_keys, 'iteration',
                trigger=(100, 'iteration'), file_name='loss.png'))

    if args.data_test_x:
        eval_dataset = test_dataset
    else:
        eval_dataset = train_dataset

    eval_interval = (args.eval_interval, 'iteration')
    trainer.extend(
        visualization(gen_g, gen_f, eval_dataset, os.path.join(args.out, 'preview'), args.vis_batch),
        trigger=eval_interval
    )

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()

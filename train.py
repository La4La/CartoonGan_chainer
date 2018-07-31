#!/usr/bin/env python

import chainer
import os

from chainer import training
from chainer.training import extensions

import argparse
from model import Generator, Discriminator, VGG
from updater import CartoonGAN
from dataset import PhotoDataset, ImageDataset

os.environ["CHAINER_TYPE_CHECK"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=12)
    parser.add_argument('--epoch', '-e', type=int, default=500)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--root', '-R', default='/mnt/sakuradata10-striped/gao/background')
    parser.add_argument('--out', '-o', default='/mnt/sakuradata10-striped/gao/results/cartoongan')
    parser.add_argument('--resume', '-r', default='', help='snapshot No.')
    parser.add_argument('--model_num', '-m', default='', help='generater No.')
    parser.add_argument('--snapshot_interval', type=int, default=1000)
    parser.add_argument('--test_interval', type=int, default=100)
    parser.add_argument('--display_interval', type=int, default=5)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--use_gan', '-G', action='store_true')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}\n'.format(args.batchsize))

    # Setup models
    gen = Generator()
    vgg = VGG()
    if args.use_gan:
        dis = Discriminator()
    else:
        dis = None

    # Setup datasets
    photos = PhotoDataset(os.path.join(args.root, 'photos_resized', '*'), crop_size=args.size)
    photos_iter = chainer.iterators.SerialIterator(photos, args.batchsize)
    if args.use_gan:
        illusts = ImageDataset(os.path.join(args.root, 'illusts', '*', '*', '*'), crop_size=args.size)
        illusts_iter = chainer.iterators.SerialIterator(illusts, args.batchsize)
    else:
        illusts_iter = None

    # models to gpu
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        vgg.to_gpu()
        if args.use_gan:
            dis.to_gpu()

    # Setup optimizer parameters.
    opt = chainer.optimizers.Adam(alpha=0.0002)
    opt.setup(gen)
    if args.use_gan:
        opt_d = chainer.optimizers.Adam(alpha=0.0002)
        opt_d.setup(dis)
    else:
        opt_d = None

    # Set up a trainer
    optimizers = {'gen': opt, 'dis': opt_d} if args.use_gan else {'gen': opt}
    iterators = {'main': photos_iter, 'illusts': illusts_iter} if args.use_gan else {'main': photos_iter}
    updater = CartoonGAN(
        models=(gen, dis, vgg),
        iterator=iterators,
        optimizer=optimizers,
        device=args.gpu,
        w=10
    )
    out = os.path.join(args.out, 'gan') if args.use_gan else os.path.join(args.out, 'initial')
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)

    # Load npz if necessary
    if args.resume:
        chainer.serializers.load_npz(os.path.join(out, 'snapshot_iter_'+args.resume+'.npz'), trainer)
        print('snapshot {} loaded\n'.format(args.resume))
    elif args.model_num:
        chainer.serializers.load_npz(os.path.join(args.out, 'initial', 'gen_iter_'+args.model_num+'.npz'), gen)
        print('model {} loaded\n'.format(args.model_num))

    # trainer extensions
    snapshot_interval = (args.snapshot_interval, 'iteration')
    test_interval = (args.test_interval, 'iteration')
    trainer.extend(extensions.dump_graph('gen/loss'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(args.display_interval, 'iteration'), ))
    report = ['epoch', 'iteration', 'gen/loss', 'gen/content', 'gen/mae']
    if args.use_gan:
        report += ['gen/adv', 'dis/illust', 'dis/edge', 'dis/photo', 'dis/loss']
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.PrintReport(report))
    trainer.extend(extensions.ProgressBar(update_interval=args.display_interval))
    trainer.extend(photos.visualizer(), trigger=test_interval)

    trainer.run()

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(args.out, 'model_final'), gen)
    chainer.serializers.save_npz(os.path.join(args.out, 'optimizer_final'), opt)


if __name__ == '__main__':
    main()


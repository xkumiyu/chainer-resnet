import os
import random

import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    raise

import chainer
from chainer import training
from chainer.training import extensions

from helpers.argparse import generate_parser
from helpers.outdir import prepare_outdir
from helpers.outdir import save_info
from models.resnet import ResNet


def main():
    parser = generate_parser(batchsize=128, epoch=200)
    parser.add_argument('--dataset', '-d', default='cifar10', choices=['cifar10', 'cifar100'],
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--n_layers', '-l', type=int,
                        default=20, choices=[20, 32, 44, 56, 110, 1202],
                        help='Number of layers for ResNet')
    args = parser.parse_args()

    # Create an outdir
    outdir = prepare_outdir(args.out, args.out_suffix)

    # Setup a dataset
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = chainer.datasets.get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = chainer.datasets.get_cifar100()

    def data_augment(data):
        image, label = data
        xp = chainer.cuda.get_array_module(image)

        # after 0 padding, image.shape = (3, 40, 40)
        image = xp.pad(image, ((0, 0), (4, 4), (4, 4)), 'constant', constant_values=(0, 0))
        _, h, w = image.shape
        crop_size = 32

        top = random.randint(0, h - crop_size - 1)
        left = random.randint(0, w - crop_size - 1)
        if random.randint(0, 1):
            image = image[:, :, ::-1]
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        return image, label
    train = chainer.datasets.TransformDataset(train, data_augment)

    # Setup iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    # Setup a model
    model = ResNet(args.n_layers, class_labels)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.MomentumSGD(0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(10e-5))

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=outdir)

    display_interval = (args.display_interval, 'iteration')
    snapshot_interval = (args.snapshot_interval, 'iteration')

    # trainer.extend(extensions.LinearShift('lr', (10e-3, 10e-4), (48000, 48001)))
    # trainer.extend(extensions.LinearShift('lr', (10e-2, 10e-3), (32000, 32001)))
    trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=(16000, 'iteration'))

    trainer.extend(extensions.Evaluator(
        test_iter, model, device=args.gpu), trigger=display_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.observe_lr(), trigger=display_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            trigger=display_interval, file_name='accuracy.png'))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            trigger=display_interval, file_name='loss.png'))
        trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Save infomation
    save_info(outdir, args, model, optimizer, train, test)

    # Display infomation
    print('GPU: {}'.format(args.gpu))
    print('Model: ResNet{}'.format(args.n_layers))
    print('Optimizer: {}'.format(optimizer.__class__.__name__))
    print('Epoch: {}'.format(args.epoch))
    print('Batch Size: {}'.format(args.batchsize))
    print('Iter per Epoch: {}'.format(int(len(train) / args.batchsize)))
    print('Train Samples: {}'.format(len(train)))
    print('Test Samples: {}'.format(len(test)))
    print('Data Shape: {}'.format(train[0][0].shape if type(train[0]) == tuple else train[0].shape))
    print('Directory to output: {}'.format(outdir))
    print('')

    # Run the training
    trainer.run()

    # Save model
    if args.gpu >= 0:
        model.to_cpu()
    chainer.serializers.save_npz(os.path.join(outdir, 'model.npz'), model)

    print('Finished!')


if __name__ == '__main__':
    main()

import argparse


def generate_parser(description=None, archs=[], **default):
    init_default = {
        'batchsize': 32,
        'epoch': 100,
        'gpu': -1,
        'out': 'result',
        'out_suffix': '',
        'snapshot_interval': 10000,
        'display_interval': 1000,
        'resume': '',
        'arch': ''
    }
    init_default.update(default)

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batchsize', '-b', type=int,
                        default=init_default['batchsize'],
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int,
                        default=init_default['epoch'],
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int,
                        default=init_default['gpu'],
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o',
                        default=init_default['out'],
                        help='Directory to output the result')
    parser.add_argument('--out_suffix', '-s',
                        default=init_default['out_suffix'])
    parser.add_argument('--snapshot_interval', type=int,
                        default=init_default['snapshot_interval'],
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int,
                        default=init_default['display_interval'],
                        help='Interval of displaying log to console')
    parser.add_argument('--resume', '-r',
                        default=init_default['resume'],
                        help='Resume the training from snapshot')
    if len(archs) > 0:
        parser.add_argument('--arch', '-a', choices=archs,
                            default=init_default['arch'],
                            help='Using architecture')
    return parser

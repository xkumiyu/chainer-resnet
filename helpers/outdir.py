import argparse
import datetime
import json
import os
import sys


def prepare_outdir(parent_ourdir, outdir_suffix='', time_format='%Y%m%dT%H%M%S'):
    time_str = datetime.datetime.now().strftime(time_format)
    if os.path.exists(parent_ourdir):
        if not os.path.isdir(parent_ourdir):
            raise RuntimeError('{} is not a directory'.format(parent_ourdir))
    if outdir_suffix:
        time_str = time_str + '_' + outdir_suffix
    outdir = os.path.join(parent_ourdir, time_str)
    if os.path.exists(outdir):
        raise RuntimeError('{} exists'.format(outdir))
    else:
        os.makedirs(outdir)
    return outdir


def chainer_info(model, optimizer, train, test):
    import chainer

    info = {}
    info['version'] = {'chainer': chainer.__version__}
    info['model'] = {link[0]: link[1].__class__.__name__ for link in model.namedlinks()}
    info['optimizer'] = {
        'name': optimizer.__class__.__name__,
        'init_param': optimizer.hyperparam.get_dict()
    }
    info['dataset'] = {
        'length': {'train': len(train)},
        'shape': train[0][0].shape if type(train[0]) == tuple else train[0].shape
    }
    if test:
        info['dataset']['length']['test'] = len(test)
    return info


def save_info(outdir, args, model, optimizer, train, test=None, argv=None):
    json_kwargs = {
        'indent': 4,
        'sort_keys': True,
        'separators': (',', ': ')
    }

    # Save all the arguments
    with open(os.path.join(outdir, 'args.json'), 'w') as f:
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        f.write(json.dumps(args, **json_kwargs))

    # Save all the environment variables
    with open(os.path.join(outdir, 'environ.json'), 'w') as f:
        f.write(json.dumps(dict(os.environ), **json_kwargs))

    # Save the command
    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        f.write(' '.join(sys.argv))

    # Save the chainer infomation
    with open(os.path.join(outdir, 'info.json'), 'w') as f:
        f.write(json.dumps(chainer_info(model, optimizer, train, test), **json_kwargs))

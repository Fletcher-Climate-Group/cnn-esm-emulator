from multiprocessing import get_context
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tf.nets import f09_model

import argparse
import yaml
import numpy as np
from time import time
import os.path as osp
import pickle
from datetime import datetime
from utils.data import get_data_lr_hr
from utils.plots import plot_multi_res_error, plot_feature_importance
from utils.losses import ss_loss
import sys


def initialize_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            print('GPU already initialized.')


def update_cfg(cfg, seed, n_hr, gpu, name=''):
    cfg['seed'] = seed
    cfg['n_hr'] = n_hr
    cfg['gpu'] = gpu
    cfg['model_name'] = name
    return cfg


def train_multi_res(cfg):
    initialize_gpus()

    np.random.seed(cfg['seed'])
    tf.random.set_seed(cfg['seed'])

    hr_shuffle = np.random.permutation(cfg['in_hr'].shape[0])
    test_ids = hr_shuffle[:cfg['n_test']]
    train_ids = hr_shuffle[cfg['n_test']:]

    test_x = cfg['in_hr'][test_ids]
    train_x_hr = cfg['in_hr'][train_ids]
    test_y = cfg['out_hr'][test_ids]
    train_y_hr = cfg['out_hr'][train_ids]

    meta = {
        'hr_test_ids': test_ids,
        'hr_train_ids': train_ids,
        'n_hr': cfg['n_hr'],
        'seed': cfg['seed']
    }

    if cfg['n_train'] > 0:
        # remove random low/med res samples to keep consistent number of training examples
        lr_shuffle = np.random.permutation(cfg['in_lr'].shape[0])
        remove = cfg['in_lr'].shape[0] - cfg['n_train'] + cfg['n_hr']
        in_lr = cfg['in_lr'][lr_shuffle[:-remove]]
        out_lr = cfg['out_lr'][lr_shuffle[:-remove]]
        train_x = np.concatenate((in_lr, train_x_hr[:cfg['n_hr']]), axis=0)
        train_y = np.concatenate((out_lr, train_y_hr[:cfg['n_hr']]), axis=0)
    else:
        train_x = np.concatenate((cfg['in_lr'], train_x_hr[:cfg['n_hr']]), axis=0)
        train_y = np.concatenate((cfg['out_lr'], train_y_hr[:cfg['n_hr']]), axis=0)

    test_y = test_y.transpose([0, 2, 3, 1])
    train_y = train_y.transpose([0, 2, 3, 1])

    strategy = tf.distribute.OneDeviceStrategy('GPU:{}'.format(cfg['gpu']))

    with strategy.scope():
        model = f09_model(train_x.shape[-1], train_y.shape[-1],
                          cfg['width_mult'], cfg['kernel_size'],
                          cfg['dropout'], cfg['double_layers'])
        train_ds = (tf.data.Dataset.from_tensor_slices((train_x, train_y))
                    .shuffle(500).repeat().batch(cfg['batch_size']))
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(cfg['batch_size'])
        spe = int(np.ceil(train_x.shape[0] // cfg['batch_size']))
        lr = tf.keras.experimental.CosineDecay(cfg['lr'], spe * cfg['epochs'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=ss_loss, metrics=["mse"])

    ts = time()
    hist = model.fit(train_ds, epochs=cfg['epochs'], steps_per_epoch=spe, validation_data=test_ds, verbose=0)
    t = time() - ts
    if cfg['save']:
        model.save(osp.join(cfg['save_dir'], cfg['model_name'] + '.h5'))
    meta['train_loss'] = hist.history['loss']
    meta['train_mse'] = hist.history['mse']
    meta['val_loss'] = hist.history['val_loss']
    meta['val_mse'] = hist.history['val_mse']
    meta['test_x'] = test_x
    meta['test_y'] = test_y.astype(np.float16)
    meta['preds'] = model.predict(test_ds).astype(np.float16)
    meta['time'] = t
    print('{} | time: {:.0f}s | SS: {:.4f}'.format(cfg['model_name'], meta['time'], 1 - meta['val_loss'][-1]))
    pickle.dump(meta, open(osp.join(cfg['save_dir'], cfg['model_name'] + '_meta.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--exp-dir', default='experiments/tf/multi-res')
    parser.add_argument('--exp-name', default='')
    parser.add_argument('--n-gpus', type=int, default=4)
    parser.add_argument('--resize', default='bilinear')
    parser.add_argument('--res-ids', nargs='+', type=float, default=[1, 1/4, 1/16])
    parser.add_argument('--low-res', nargs='+', default=['f45', 'f19'])
    parser.add_argument('--high-res', default='f09')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--n-train', type=int, default=-1)
    parser.add_argument('--n-hr', type=int, nargs='+',
                        default=[0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80])
    parser.add_argument('--n-trials', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--width-mult', type=float, default=1.)
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--double-layers', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--nhr-feat', type=int, nargs='+', default=[40])
    args = parser.parse_args()

    exp_dir = osp.join(args.exp_dir, args.exp_name if args.exp_name else datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(exp_dir)

    assert args.n_train <= 200
    assert args.n_trials % args.n_gpus == 0
    assert len(args.res_ids) == len(args.low_res + [args.high_res])
    if args.plot:
        for nhr_feat in args.nhr_feat:
            assert nhr_feat in args.n_hr

    initialize_gpus()

    with open(osp.join(exp_dir, 'args.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f)

    in_lr, out_lr, in_hr, out_hr = get_data_lr_hr(
        args.data_dir, args.low_res, args.high_res, args.resize, args.res_ids)

    train_cfg = {
        'in_lr': in_lr,
        'in_hr': in_hr,
        'out_lr': out_lr,
        'out_hr': out_hr,
        'n_test': args.n_test,
        'n_train': args.n_train,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'width_mult': args.width_mult,
        'kernel_size': args.kernel_size,
        'dropout': args.dropout,
        'double_layers': args.double_layers,
        'save_dir': exp_dir,
        'save': args.save_models
    }

    print('Training models...')
    for i in range(len(args.n_hr)):
        for j in range(args.n_trials // args.n_gpus):
            cfgs = []
            for k in range(args.n_gpus):
                seed = i * args.n_trials + j * args.n_gpus + k
                n_hr = args.n_hr[i]
                cfgs.append(update_cfg(train_cfg.copy(), seed, n_hr=n_hr, gpu=k,
                                       name='nhr{}_t{}'.format(n_hr, j * args.n_gpus + k)))
            with get_context("spawn").Pool(args.n_gpus) as p:
                p.map(train_multi_res, cfgs)

    if args.plot:
        print('Plotting Mean Error Maps...')
        plot_multi_res_error(exp_dir, out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'])

        if args.save_models:
            print('Plotting Feature Importance Scores...')
            for nhr_feat in args.nhr_feat:
                plot_feature_importance(exp_dir, nhr_feat,
                                        out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'],
                                        ext='png')

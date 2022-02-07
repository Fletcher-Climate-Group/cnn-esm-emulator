import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add project folder to path

import tensorflow as tf
from tensorflow.keras import layers
import argparse
from utils.data import load_data
import numpy as np
import os
import os.path as osp
from datetime import datetime
import yaml
from time import time
import pickle
from utils.plots import plot_predictions, plot_losses
from utils.losses import ss_loss
from tf.nets import f45_model, f19_model, f09_model

import tempfile
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph


def get_flops(model, write_path=tempfile.NamedTemporaryFile().name):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        if write_path:
            opts['output'] = 'file:outfile={}'.format(write_path)  # suppress output
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops / 2  # divide by two for fused multiply-accumulate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--exp-dir', default='experiments/tf/single-res')
    parser.add_argument('--exp-name', default='')
    parser.add_argument('--res', default='f09', help='one of f09, f19, or f45')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--width-mult', type=float, default=1.)
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--double-layers', action='store_true')
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--loss', default='ss')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    assert args.res in ['f09', 'f19', 'f45']
    model_dict = {'f09': f09_model, 'f19': f19_model, 'f45': f45_model}
    assert args.kernel_size in [1, 3, 5, 7], 'kernel size must be one of [1, 3, 5, 7]'

    exp_dir = osp.join(args.exp_dir, args.exp_name if args.exp_name else datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # exp_dir += '_{}'.format(args.res)
    if osp.exists(exp_dir):
        print('{} exists already'.format(exp_dir))
        sys.exit()
    os.makedirs(exp_dir)

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    inputs, outputs, norm_inputs, norm_outputs, parent_maps = load_data(args.data_dir, args.res, 'sstref')

    shuffle = np.random.permutation(inputs.shape[0])
    test_idx = shuffle[:args.n_test]
    train_idx = shuffle[args.n_test:]

    train_x, train_y = norm_inputs[train_idx], norm_outputs[train_idx]
    test_x, test_y = norm_inputs[test_idx], norm_outputs[test_idx]

    print('train x shape:', train_x.shape)
    print('train y shape:', train_y.shape)
    print('test x shape:', test_x.shape)
    print('test y shape:', test_y.shape)

    with open(osp.join(exp_dir, 'args.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f)

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(100).repeat().batch(args.batch)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batch)

    spe = int(np.ceil(train_x.shape[0] // args.batch))
    lr = tf.keras.experimental.CosineDecay(args.lr, spe * args.epochs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model = model_dict[args.res](train_x.shape[-1], train_y.shape[-1],
                                 w=args.width_mult, k=args.kernel_size,
                                 dr=args.dropout, double=args.double_layers)
    model.summary()

    if args.loss == 'ss':
        loss_object = ss_loss
    else:
        loss_object = tf.keras.losses.MSE

    model.compile(optimizer=optimizer, loss=loss_object, metrics=["mse"])
    ts = time()
    hist = model.fit(train_ds, epochs=args.epochs, steps_per_epoch=spe, validation_data=test_ds)
    t = time() - ts
    print('Training took {:d} seconds'.format(int(t)))
    print('Validation Loss: {:.6}'.format(hist.history['val_loss'][-1]))
    print('Validation MSE: {:.6f}'.format(hist.history['val_mse'][-1]))

    model.save(osp.join(exp_dir, 'model.h5'))

    meta = {'train_loss': hist.history['loss'],
            'train_mse': hist.history['mse'],
            'val_loss': hist.history['val_loss'],
            'val_mse': hist.history['val_mse'],
            'test_idx': test_idx,
            'training_time': t,
            'parameters': model.count_params(),
            'operations': get_flops(model)}

    pickle.dump(meta, open(osp.join(exp_dir, 'meta.pkl'), 'wb'))

    if args.plot:
        preds = model.predict(test_x)
        print('Making plots...')
        plot_dir = osp.join(exp_dir, 'plots')
        plot_predictions(x=preds[:1],  # first test sample
                         plot_dir=plot_dir,
                         sample_ids=test_idx[:1],
                         gt=test_y[:1],
                         out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'])

        plot_losses(meta['train_loss'], meta['val_loss'], plot_dir, loss_name=args.loss, start_idx=10, title='TensorFlow')
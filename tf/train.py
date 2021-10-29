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
from utils.plots import plot_predictions
from utils.loss import ss_loss


def f09_model(in_c, out_c):
    model = tf.keras.Sequential()

    model.add(layers.Dense(6*9*256, use_bias=False, input_shape=(in_c,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((6, 9, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(out_c, (5, 5), strides=(2, 2), padding='same'))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--exp-dir', default='tf/experiments/single-res')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--loss', default='ss')
    args = parser.parse_args()

    np.random.seed(0)
    tf.random.set_seed(0)

    inputs, outputs, norm_inputs, norm_outputs, parent_maps = load_data(args.data_dir, 'f09', 'sstref')

    shuffle = np.random.permutation(inputs.shape[0])
    test_idx = shuffle[:args.n_test]
    train_idx = shuffle[args.n_test:]

    train_x, train_y = norm_inputs[train_idx], norm_outputs[train_idx]
    test_x, test_y = norm_inputs[test_idx], norm_outputs[test_idx]

    print('train x shape:', train_x.shape)
    print('train y shape:', train_y.shape)
    print('test x shape:', test_x.shape)
    print('test y shape:', test_y.shape)

    exp_dir = osp.join(args.exp_dir, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(exp_dir)

    with open(osp.join(exp_dir, 'args.yaml'), 'w') as f:
        yaml.safe_dump(vars(args), f)

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(100).repeat().batch(args.batch)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(args.batch)

    spe = int(np.ceil(train_x.shape[0] // args.batch))
    lr = tf.keras.experimental.CosineDecay(args.lr, spe * args.epochs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model = f09_model(train_x.shape[-1], train_y.shape[-1])
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
            'training_time': t}

    pickle.dump(meta, open(osp.join(exp_dir, 'meta.pkl'), 'wb'))

    preds = model.predict(test_x)

    print('Plotting predictions...')
    # first test sample only
    plot_predictions(x=preds[:1],
                     plot_dir=osp.join(exp_dir, 'plots'),
                     sample_ids=test_idx[:1],
                     gt=test_y[:1],
                     out_names=['AOD', 'CLDL', 'FNET', 'LWCF', 'PRECT', 'QRL', 'SWCF'])


import tensorflow as tf
from tensorflow.keras import layers


def f09_model(in_c=9, out_c=7, w=1, k=5, dr=0, double=False):
    """
    :param in_c: input channels (9 ESM parameters)
    :param out_c: output channels (7 ESM variables)
    :param w: width multiplier (scale base number of filters)
    :param k: kernel size
    :param dr: dropout rate
    :param double: double layers
    :return: f09 CNN
    """
    model = tf.keras.Sequential()

    model.add(layers.Dense(6*9*int(256*w), use_bias=False, input_shape=(in_c,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    model.add(layers.Reshape((6, 9, int(256*w))))

    model.add(layers.Conv2DTranspose(int(128*w), k, strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(128 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(64*w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(64 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(64*w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(64 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(32*w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(32 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(32*w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(32 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(out_c, k, strides=(2, 2), padding='same'))

    return model


def f19_model(in_c=9, out_c=7, w=1, k=5, dr=0, double=False):
    """
    :param in_c: input channels (9 ESM parameters)
    :param out_c: output channels (7 ESM variables)
    :param w: width multiplier (scale base number of filters)
    :param k: kernel size
    :param dr: dropout rate
    :param double: double layers
    :return: f19 CNN
    """
    model = tf.keras.Sequential()

    model.add(layers.Dense(6*9*int(256*w), use_bias=False, input_shape=(in_c,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    model.add(layers.Reshape((6, 9, int(256*w))))

    model.add(layers.Conv2DTranspose(int(128*w), k, strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(128 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(64*w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(64 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(64*w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(64 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(32*w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(32 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(out_c, k, strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 96, 144, out_c)

    return model


def f45_model(in_c=9, out_c=7, w=1, k=5, dr=0, double=False):
    """
    :param in_c: input channels (9 ESM parameters)
    :param out_c: output channels (7 ESM variables)
    :param w: width multiplier (scale base number of filters)
    :param k: kernel size
    :param dr: dropout rate
    :param double: double layers
    :return: f19 CNN
    """
    model = tf.keras.Sequential()

    model.add(layers.Dense(6 * 9 * int(256 * w), use_bias=False, input_shape=(in_c,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    model.add(layers.Reshape((6, 9, int(256 * w))))

    model.add(layers.Conv2DTranspose(int(128 * w), k, strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(128 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(64 * w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(64 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(int(32 * w), k, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dr))

    if double:
        model.add(layers.Conv2DTranspose(int(32 * w), k, strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(dr))

    model.add(layers.Conv2DTranspose(out_c, k, strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 48, 72, out_c)

    return model


if __name__ == '__main__':
    model = f45_model()
    model.summary()
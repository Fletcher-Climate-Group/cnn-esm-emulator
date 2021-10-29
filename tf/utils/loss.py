import tensorflow as tf


@tf.function
def ss_loss(gt, pred, reduce_mean=True):
    gt_mean = tf.reduce_mean(gt, axis=[1, 2], keepdims=True)
    MSE = tf.reduce_mean((gt-pred)**2, axis=[1, 2])
    MSE_norm = tf.reduce_mean((gt_mean - pred)**2, axis=[1, 2])
    if reduce_mean:
        return tf.reduce_mean(MSE / MSE_norm)
    else:
        return MSE / MSE_norm
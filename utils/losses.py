import tensorflow as tf
import torch
from torch.distributions import Normal

@tf.function
def ss_loss(gt, pred, reduce_mean=True):
    gt_mean = tf.reduce_mean(gt, axis=[1, 2], keepdims=True)
    MSE = tf.reduce_mean((gt-pred)**2, axis=[1, 2])
    MSE_norm = tf.reduce_mean((gt_mean - pred)**2, axis=[1, 2])
    if reduce_mean:
        return tf.reduce_mean(MSE / MSE_norm)
    else:
        return MSE / MSE_norm


def probabilistic_loss(targets, sigma, mu):
    # log_lik_loss = F.mse_loss(mu, targets, reduction='mean')
    p_pred = Normal(mu, sigma)
    log_prob = p_pred.log_prob(targets)
    log_lik_loss = -torch.mean(log_prob)
    return log_lik_loss
import tensorflow as tf
import torch
from torch.distributions import Normal


def probabilistic_loss(targets, sigma, mu):
    # log_lik_loss = F.mse_loss(mu, targets, reduction='mean')
    p_pred = Normal(mu, sigma)
    log_prob = p_pred.log_prob(targets)
    log_lik_loss = -torch.mean(log_prob)
    return log_lik_loss
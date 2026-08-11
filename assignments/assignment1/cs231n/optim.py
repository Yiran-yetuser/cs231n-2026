import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################

    # 新速度 = 旧速度按 momentum 衰减 + 当前负梯度步长
    v = config["momentum"] * v - config["learning_rate"] * dw
    # 沿速度方向更新权重
    next_w = w + v

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################

    # 更新cache
    config["cache"] = config["decay_rate"] * config["cache"] + (
        1 - config["decay_rate"]
    ) * (dw**2)

    # 更新w（用更新后的 cache 计算自适应步长）
    next_w = w - dw * config["learning_rate"] / (
        np.sqrt(config["cache"]) + config["epsilon"]
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    ###########################################################################

    t = config['t'] + 1
    m, v = config['m'], config['v']
    beta1, beta2 = config['beta1'], config['beta2']
    lr, eps = config['learning_rate'], config['epsilon']

    # 更新移动平均
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw**2)

    # 偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # 更新参数
    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    # 保存状态
    config['m'], config['v'], config['t'] = m, v, t

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adamw(w, dw, config=None):
    """
    AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    The difference from Adam is that weight decay is NOT folded into the
    gradient; instead it is applied directly to the weights every step,
    so that the adaptive per-parameter learning rates of Adam do not
    interfere with the regularization strength.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - weight_decay: Scalar giving the decoupled L2 penalty (e.g. 5e-4).
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("weight_decay", 5e-4)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    t = config["t"] + 1
    beta1, beta2 = config["beta1"], config["beta2"]
    lr, eps = config["learning_rate"], config["epsilon"]

    # 与 Adam 的区别在于，weight decay 不参与梯度计算，而是直接作用于权重
    wd = config["weight_decay"]

    m = beta1 * config["m"] + (1 - beta1) * dw
    v = beta2 * config["v"] + (1 - beta2) * (dw**2)

    # 偏差修正
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # 更新参数：Adam 的自适应步长 + 直接作用于权重的衰减项 wd * w
    next_w = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * w)

    # 保存状态
    config["m"], config["v"], config["t"] = m, v, t

    return next_w, config

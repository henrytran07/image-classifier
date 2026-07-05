import numpy as np
from .activations import relu, softmax


def init_linear_params():
    params = {}
    params['W1'] = np.random.randn(512, 784) * np.sqrt(2 / 784)
    params['b1'] = np.zeros((512, 1))
    params['W2'] = np.random.randn(256, 512) * np.sqrt(2 / 512)
    params['b2'] = np.zeros((256, 1))
    params['W3'] = np.random.randn(128, 256) * np.sqrt(2 / 256)
    params['b3'] = np.zeros((128, 1))
    params['W4'] = np.random.randn(10, 128) * np.sqrt(2 / 128)
    params['b4'] = np.zeros((10, 1))
    return params


def init_bn_params():
    bn_params = {}
    for l, n in zip((1, 2, 3), (512, 256, 128)):
        bn_params['gamma' + str(l)] = np.ones((n, 1))
        bn_params['beta' + str(l)] = np.zeros((n, 1))
    return bn_params


def init_bn_running_stats():
    """Running mean/var used at inference time (kept separate from gamma/beta
    so that len(bn_params)//2 layer-counting stays valid)."""
    running = {}
    for l, n in zip((1, 2, 3), (512, 256, 128)):
        running['running_mean' + str(l)] = np.zeros((n, 1))
        running['running_var' + str(l)] = np.ones((n, 1))
    return running


def init_adam_state(bn_params: dict, linear_params: dict):
    adam_state = {}
    num_of_linear_layers = len(linear_params) // 2
    for l in range(1, num_of_linear_layers + 1):
        adam_state['v_dW' + str(l)] = np.zeros(linear_params['W' + str(l)].shape)
        adam_state['s_dW' + str(l)] = np.zeros(linear_params['W' + str(l)].shape)
        adam_state['v_db' + str(l)] = np.zeros(linear_params['b' + str(l)].shape)
        adam_state['s_db' + str(l)] = np.zeros(linear_params['b' + str(l)].shape)

    num_of_bn_layers = len(bn_params) // 2
    for l in range(1, num_of_bn_layers + 1):
        adam_state['v_dgamma' + str(l)] = np.zeros(bn_params['gamma' + str(l)].shape)
        adam_state['s_dgamma' + str(l)] = np.zeros(bn_params['gamma' + str(l)].shape)
        adam_state['v_dbeta' + str(l)] = np.zeros(bn_params['beta' + str(l)].shape)
        adam_state['s_dbeta' + str(l)] = np.zeros(bn_params['beta' + str(l)].shape)
    return adam_state


def batch_norm_forward(u, gamma, beta, running_mean, running_var,
                       training, momentum=0.9, epsilon=1e-8):
    """
        u: (n, m) pre-activation (W @ A_prev + b)
        gamma, beta, running_mean, running_var: (n, 1)
        Returns z_norm (n,m) and a cache for backward.
        In training mode, running_mean/var are updated IN PLACE.
    """
    if training:
        mu = u.mean(axis=1, keepdims=True)            
        var = u.var(axis=1, keepdims=True)            
        inv_std = 1.0 / np.sqrt(var + epsilon)
        z_hat = (u - mu) * inv_std

        running_mean *= momentum
        running_mean += (1 - momentum) * mu
        running_var *= momentum
        running_var += (1 - momentum) * var
    else:
        inv_std = 1.0 / np.sqrt(running_var + epsilon)
        z_hat = (u - running_mean) * inv_std
    z_norm = gamma * z_hat + beta
    cache = (z_hat, inv_std, gamma)
    return z_norm, cache


def batch_norm_backward(dz_norm, cache):
    """
        dz_norm: (n, m) gradient flowing into the BN output (already through ReLU),
                 in the same un-averaged convention the rest of the code uses.
        Returns:
            du:     (n, m) gradient w.r.t. the pre-normalization input u
            dgamma: (n, 1) NOT yet divided by m (caller applies 1/m)
            dbeta:  (n, 1) NOT yet divided by m
    """
    z_hat, inv_std, gamma = cache
    dgamma = np.sum(dz_norm * z_hat, axis=1, keepdims=True)
    dbeta = np.sum(dz_norm, axis=1, keepdims=True)
    g = dz_norm * gamma
    du = inv_std * (g
                    - g.mean(axis=1, keepdims=True)
                    - z_hat * (g * z_hat).mean(axis=1, keepdims=True))
    return du, dgamma, dbeta


def dropout(A: np.ndarray, keep_prob=0.8):
    mask = (np.random.rand(*A.shape) < keep_prob)
    A = A * mask / keep_prob
    return A, mask


def forward_prop(params, bn_params, X, training=True, keep_prob=0.8,
                 bn_running=None, momentum=0.9):
    """
        Returns (cache, A4). cache stores everything backward needs, including
        the per-layer BN caches under keys 'bn1', 'bn2', 'bn3'.
        bn_running: dict of running_mean{l}/running_var{l}. Required for a
        correct inference pass; if None, batch statistics are used.
    """
    if bn_running is None:
        bn_running = init_bn_running_stats()

    cache = {}
    A_prev = X
    for l in (1, 2, 3):
        W, b = params['W' + str(l)], params['b' + str(l)]
        gamma, beta = bn_params['gamma' + str(l)], bn_params['beta' + str(l)]
        u = W @ A_prev + b
        z_norm, bn_cache = batch_norm_forward(
            u, gamma, beta,
            bn_running['running_mean' + str(l)], bn_running['running_var' + str(l)],
            training, momentum)
        A = relu(z_norm)
        if training:
            A, mask = dropout(A, keep_prob)
            cache['mask' + str(l)] = mask
        cache['z' + str(l)] = z_norm
        cache['A' + str(l)] = A
        cache['bn' + str(l)] = bn_cache
        A_prev = A

    z4 = params['W4'] @ A_prev + params['b4']
    A4 = softmax(z4)
    cache['z4'] = z4
    return cache, A4


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def backward_prop(A4, X, Y, cache, params, keep_prob=0.8):
    grads = {}
    m = A4.shape[1]

    A3 = cache['A3']
    dZ4 = A4 - Y
    grads['dW4'] = (1 / m) * dZ4 @ A3.T
    grads['db4'] = (1 / m) * np.sum(dZ4, axis=1, keepdims=True)

    dA = params['W4'].T @ dZ4
    A_prevs = {3: cache['A2'], 2: cache['A1'], 1: X}
    for l in (3, 2, 1):
        if 'mask' + str(l) in cache:
            dA = dA * cache['mask' + str(l)] / keep_prob
        dz_norm = dA * relu_derivative(cache['z' + str(l)])
        du, dgamma, dbeta = batch_norm_backward(dz_norm, cache['bn' + str(l)])
        A_prev = A_prevs[l]
        grads['dW' + str(l)] = (1 / m) * du @ A_prev.T
        grads['db' + str(l)] = (1 / m) * np.sum(du, axis=1, keepdims=True)
        grads['dgamma' + str(l)] = (1 / m) * dgamma
        grads['dbeta' + str(l)] = (1 / m) * dbeta
        if l > 1:
            dA = params['W' + str(l)].T @ du
    return grads


def adam_optimizer(grads, adam_state, t, beta1=0.9, beta2=0.999):
    corrected = {}
    for l in range(1, 5):
        for p in ('W', 'b'):
            key = 'd' + p + str(l)
            adam_state['v_' + key] = beta1 * adam_state['v_' + key] + (1 - beta1) * grads[key]
            adam_state['s_' + key] = beta2 * adam_state['s_' + key] + (1 - beta2) * grads[key] ** 2
            corrected['v_' + key] = adam_state['v_' + key] / (1 - beta1 ** t)
            corrected['s_' + key] = adam_state['s_' + key] / (1 - beta2 ** t)
    for l in range(1, 4):
        for p in ('gamma', 'beta'):
            key = 'd' + p + str(l)
            adam_state['v_' + key] = beta1 * adam_state['v_' + key] + (1 - beta1) * grads[key]
            adam_state['s_' + key] = beta2 * adam_state['s_' + key] + (1 - beta2) * grads[key] ** 2
            corrected['v_' + key] = adam_state['v_' + key] / (1 - beta1 ** t)
            corrected['s_' + key] = adam_state['s_' + key] / (1 - beta2 ** t)
    return adam_state, corrected


def update_adam(params, bn_params, corrected, learning_rate=1e-3, epsilon=1e-8):
    for l in range(1, len(params) // 2 + 1):
        params['W' + str(l)] -= learning_rate * corrected['v_dW' + str(l)] / (np.sqrt(corrected['s_dW' + str(l)]) + epsilon)
        params['b' + str(l)] -= learning_rate * corrected['v_db' + str(l)] / (np.sqrt(corrected['s_db' + str(l)]) + epsilon)
    for l in range(1, len(bn_params) // 2 + 1):
        bn_params['gamma' + str(l)] -= learning_rate * corrected['v_dgamma' + str(l)] / (np.sqrt(corrected['s_dgamma' + str(l)]) + epsilon)
        bn_params['beta' + str(l)] -= learning_rate * corrected['v_dbeta' + str(l)] / (np.sqrt(corrected['s_dbeta' + str(l)]) + epsilon)
    return params, bn_params


def get_mini_batches(shuffled_X, shuffled_Y, mini_batch_size=64):
    m = shuffled_X.shape[1]
    num_batches = m // mini_batch_size
    mini_batches = []
    for i in range(num_batches):
        s, e = i * mini_batch_size, (i + 1) * mini_batch_size
        mini_batches.append((shuffled_X[:, s:e], shuffled_Y[:, s:e]))
    if m % mini_batch_size != 0:
        s = num_batches * mini_batch_size
        mini_batches.append((shuffled_X[:, s:], shuffled_Y[:, s:]))
    return mini_batches

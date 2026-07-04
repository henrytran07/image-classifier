import numpy as np
from .activations import relu, softmax, z_score_normalization


def init_linear_params():
    """
        W1: (512, 784), b1: (512, 1)
        W2: (256, 512), b2: (256, 1)
        W3: (128, 256), b3: (128, 1)
        W4: (10, 128),  b4: (10, 1)
        Return:
            dict() contains W1, W2, W3, W4, b1, b2, b3, b4
    """
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
    """
        gamma1 and beta1: (512, 1)
        gamma2 and beta2: (256, 1)
        gamma3 and beta3: (128, 1)
    """
    bn_params = {}
    bn_params['gamma1'] = np.ones((512, 1))
    bn_params['beta1']  = np.zeros((512, 1))

    bn_params['gamma2'] = np.ones((256, 1))
    bn_params['beta2']  = np.zeros((256, 1))

    bn_params['gamma3'] = np.ones((128, 1))
    bn_params['beta3']  = np.zeros((128, 1))

    return bn_params


def init_adam_state(bn_params: dict, linear_params: dict):
    """
        Return:
            adam_state: dict with v and s for each W, b, gamma, beta
    """
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


def batch_normalization(gamma, beta, z_hat):
    """
        gamma and beta: (n, 1)
        z_hat: (n, m)
        Return:
            z_norm: (n, m)
    """
    return gamma * z_hat + beta


def forward_prop(params: dict, bn_params: dict, X: np.ndarray):
    """
        params: W1, b1, W2, b2, W3, b3, W4, b4
        X: (784, m)
        Return:
            cache: dict contains z1, A1, z2, A2, z3, A3, z4
            A4: (10, m)
    """
    cache = {}

    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    W4, b4 = params['W4'], params['b4']

    gamma1, beta1 = bn_params['gamma1'], bn_params['beta1']
    gamma2, beta2 = bn_params['gamma2'], bn_params['beta2']
    gamma3, beta3 = bn_params['gamma3'], bn_params['beta3']

    z1_hat = z_score_normalization(W1 @ X + b1)
    z1_norm = batch_normalization(gamma1, beta1, z1_hat)
    A1 = relu(z1_norm)
    cache['z1'], cache['A1'] = z1_norm, A1

    z2_hat = z_score_normalization(W2 @ A1 + b2)
    z2_norm = batch_normalization(gamma2, beta2, z2_hat)
    A2 = relu(z2_norm)
    cache['z2'], cache['A2'] = z2_norm, A2

    z3_hat = z_score_normalization(W3 @ A2 + b3)
    z3_norm = batch_normalization(gamma3, beta3, z3_hat)
    A3 = relu(z3_norm)
    cache['z3'], cache['A3'] = z3_norm, A3

    z4 = W4 @ A3 + b4
    A4 = softmax(z4)
    cache['z4'] = z4

    return cache, A4


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
        Returns 1 where z > 0, else 0
    """
    return (z > 0).astype(float)


def backward_prop(A4: np.ndarray, X: np.ndarray, Y: np.ndarray, cache: dict, params: dict):
    """
        A4: (10, m)
        X:  (784, m)
        Y:  (10, m)  one-hot
        cache: z1, A1, z2, A2, z3, A3, z4
        params: W1, b1, W2, b2, W3, b3, W4, b4
        Return:
            grads: dW1, db1, dgamma1, dbeta1, ..., dW4, db4
    """
    grads = {}
    m = A4.shape[1]

    z1, A1 = cache['z1'], cache['A1']
    z2, A2 = cache['z2'], cache['A2']
    z3, A3 = cache['z3'], cache['A3']

    dZ4 = A4 - Y
    grads['dW4'] = (1 / m) * dZ4 @ A3.T
    grads['db4'] = (1 / m) * np.sum(dZ4, axis=1, keepdims=True)

    dZ3 = params['W4'].T @ dZ4 * relu_derivative(z3)
    grads['dW3']     = (1 / m) * dZ3 @ A2.T
    grads['db3']     = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    grads['dgamma3'] = (1 / m) * np.sum(dZ3 * z3, axis=1, keepdims=True)
    grads['dbeta3']  = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = params['W3'].T @ dZ3 * relu_derivative(z2)
    grads['dW2']     = (1 / m) * dZ2 @ A1.T
    grads['db2']     = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    grads['dgamma2'] = (1 / m) * np.sum(dZ2 * z2, axis=1, keepdims=True)
    grads['dbeta2']  = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = params['W2'].T @ dZ2 * relu_derivative(z1)
    grads['dW1']     = (1 / m) * dZ1 @ X.T
    grads['db1']     = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads['dgamma1'] = (1 / m) * np.sum(dZ1 * z1, axis=1, keepdims=True)
    grads['dbeta1']  = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return grads


def adam_optimizer(grads: dict, adam_state: dict, t: int, beta1=0.9, beta2=0.999):
    """
        Updates adam_state with corrected v and s moments.
        grads: dW1, db1, dgamma1, dbeta1, ..., dW4, db4
        t: current step (for bias correction)
        Return:
            adam_state with corrected moments
    """
    corrected = {}

    num_of_linear_layers = 4
    for l in range(1, num_of_linear_layers + 1):
        # W moments
        adam_state['v_dW' + str(l)] = beta1 * adam_state['v_dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        adam_state['s_dW' + str(l)] = beta2 * adam_state['s_dW' + str(l)] + (1 - beta2) * grads['dW' + str(l)]**2
        corrected['v_dW' + str(l)] = adam_state['v_dW' + str(l)] / (1 - beta1**t)
        corrected['s_dW' + str(l)] = adam_state['s_dW' + str(l)] / (1 - beta2**t)

        # b moments
        adam_state['v_db' + str(l)] = beta1 * adam_state['v_db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
        adam_state['s_db' + str(l)] = beta2 * adam_state['s_db' + str(l)] + (1 - beta2) * grads['db' + str(l)]**2
        corrected['v_db' + str(l)] = adam_state['v_db' + str(l)] / (1 - beta1**t)
        corrected['s_db' + str(l)] = adam_state['s_db' + str(l)] / (1 - beta2**t)

    num_of_bn_layers = 3
    for l in range(1, num_of_bn_layers + 1):
        # gamma moments
        adam_state['v_dgamma' + str(l)] = beta1 * adam_state['v_dgamma' + str(l)] + (1 - beta1) * grads['dgamma' + str(l)]
        adam_state['s_dgamma' + str(l)] = beta2 * adam_state['s_dgamma' + str(l)] + (1 - beta2) * grads['dgamma' + str(l)]**2
        corrected['v_dgamma' + str(l)] = adam_state['v_dgamma' + str(l)] / (1 - beta1**t)
        corrected['s_dgamma' + str(l)] = adam_state['s_dgamma' + str(l)] / (1 - beta2**t)

        # beta moments
        adam_state['v_dbeta' + str(l)] = beta1 * adam_state['v_dbeta' + str(l)] + (1 - beta1) * grads['dbeta' + str(l)]
        adam_state['s_dbeta' + str(l)] = beta2 * adam_state['s_dbeta' + str(l)] + (1 - beta2) * grads['dbeta' + str(l)]**2
        corrected['v_dbeta' + str(l)] = adam_state['v_dbeta' + str(l)] / (1 - beta1**t)
        corrected['s_dbeta' + str(l)] = adam_state['s_dbeta' + str(l)] / (1 - beta2**t)

    return adam_state, corrected


def update_adam(params: dict, bn_params: dict, corrected: dict, learning_rate=1e-3, epsilon=1e-8):
    """
        Updates W, b, gamma, beta using corrected Adam moments.
        params: W1, b1, ..., W4, b4
        bn_params: gamma1, beta1, ..., gamma3, beta3
        corrected: bias-corrected v and s from adam_optimizer
    """
    num_of_linear_layers = len(params) // 2
    for l in range(1, num_of_linear_layers + 1):
        params['W' + str(l)] -= learning_rate * corrected['v_dW' + str(l)] / (np.sqrt(corrected['s_dW' + str(l)]) + epsilon)
        params['b' + str(l)] -= learning_rate * corrected['v_db' + str(l)] / (np.sqrt(corrected['s_db' + str(l)]) + epsilon)

    num_of_bn_layers = len(bn_params) // 2
    for l in range(1, num_of_bn_layers + 1):
        bn_params['gamma' + str(l)] -= learning_rate * corrected['v_dgamma' + str(l)] / (np.sqrt(corrected['s_dgamma' + str(l)]) + epsilon)
        bn_params['beta' + str(l)]  -= learning_rate * corrected['v_dbeta' + str(l)]  / (np.sqrt(corrected['s_dbeta' + str(l)])  + epsilon)

    return params, bn_params
import numpy as np 
from .activations import relu, softmax, z_score_normalization
def init_linear_params(): 
    """
        W1: (n1, 784), b1: (n1, 1)
        W2: (n2, n1), b2: (n2, 1)
        W3: (n2, n1), b3: (n2, 1)
        W4: (n_classes, n3), b4, (n_classes, 1)
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

def init_bn_parmas(): 
    """
        gamma1 and beta1: (512, 1)
        gamma2 and beta2: (256, 1)
        gamma3 and beta3: (128, 1)
    """
    bn_params = {}
    bn_params['gamma1'] = np.ones((512, 1))
    bn_params['beta1'] = np.zeros((512, 1))

    bn_params['gamma2'] = np.ones((256, 1))
    bn_params['beta2'] = np.zeros((256, 1))

    bn_params['gamma3'] = np.ones((128, 1))
    bn_params['beta3'] = np.zeros((128, 1))
    return bn_params

def batch_normalization(gamma, beta, z_hat): 
    """
        gamma and beta: (n, 1)
        z_hat: (n, m)
        Return: 
            z_norm (n, m)
    """
    return gamma * z_hat + beta


def forward_prop(params, bn_params, X): 
    """ 
        params: W1, b1, W2, b2, W3, b3, W4, b4
        X: (784, m)
        z1: (n1, m), A1: (n1, m)
        z2: (n2, m), A2: (n2, m)
        z3: (n3, m), A3: (n3, m)
        z4: (n4, m), A4: (n4, m)

        bn_params: gamma1, beta1, gamma2, beta2, gamma3, beta3
        Return: 
            cache: dict contains z4, A3, z3, A2, z2, A1, z1
            A4: (n_classes, m)
    """ 
    cache = {}
    W1, b1, = params['W1'], params['b1']
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

def relu_derivative(z : np): 
    """
        if z > 0: da / dz = 1 
        or da / az = 0 if z <= 0
    """ 
    def filter_function(z : np): 
        return z > 0

    f = filter_function(z)
    return f.astype(float)

def backward_prop(A4 : np, X: np, Y: np, cache : dict, params: dict): 
    """
        A4: (n_classes, m)
        cache: dict contains z1, A1, z2, A2, z3, A3, z4
        params: dict contains W1, b1, W2, b2, W3, b3, W4, b4
        Y: ground_truth (1, m)
        Return: 
            params: dict contains dW1, db1, dW2, db2, dW3, db3, dW4, db4

    """
    grads = {}
    m = A4.shape[1]
    z1, A1 = cache['z1'], cache['A1']
    z2, A2 = cache['z2'], cache['A2']
    z3, A3 = cache['z3'], cache['A3']

    dZ4 = A4 - Y
    dW4 = (1 / m) * dZ4 @ A3.T
    db4 = (1 / m) * np.sum(dZ4, axis=1, keepdims=True)
    
    dZ3 = params['W4'].T @ dZ4 * relu_derivative(z3)
    dW3 = (1/ m) * dZ3 @ A2.T
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = params['W3'].T @ dZ3 * relu_derivative(z2)
    dW2 = (1/ m) * dZ2 @ A1.T
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = params['W2'].T @ dZ2 * relu_derivative(z1)
    dW1 = (1 / m) * dZ1 @ X.T
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads['dW4'], grads['db4'] = dW4, db4
    grads['dW3'], grads['db3'] = dW3, db3
    grads['dW2'], grads['db2'] = dW2, db2
    grads['dW1'], grads['db1'] = dW1, db1
    
    return grads






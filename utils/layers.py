import numpy as np 
from .activations import relu, softmax
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

def forward_prop(params, X): 
    """ 
        params: W1, b1, W2, b2, W3, b3, W4, b4
        X: (784, m)
        z1: (n1, m), A1: (n1, m)
        z2: (n2, m), A2: (n2, m)
        z3: (n3, m), A3: (n3, m)
        z4: (n4, m), A4: (n4, m)
        Return: 
            cache: dict contains z4, A3, z3, A2, z2, A1, z1
            A4: (n_classes, m)
    """ 
    cache = {}
    W1, b1, = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    W4, b4 = params['W4'], params['b4']

    z1 = W1 @ X + b1 
    A1 = relu(z1)
    cache['z1'], cache['A1'] = z1, A1

    z2 = W2 @ A1 + b2 
    A2 = relu(z2)
    cache['z2'], cache['A2'] = z2, A2

    z3 = W3 @ A2 + b3 
    A3 = relu(z3)
    cache['z3'], cache['A3'] = z3, A3

    z4 = W4 @ A3 + b4 
    A4 = softmax(z4)
    cache['z4'] = z4 

    return cache, A4 

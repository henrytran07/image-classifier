import numpy as np 

def z_score_normalization(x : np, epsilon = 1e-8):
    """
        np.array(x)
        Return:  
            np.array(x_normalized)
    """ 
    return (x - x.mean()) / np.sqrt(x.std()**2 + epsilon) 

def relu(z):
    """
        Return: 
            relu(z): shape(n, m) 
            relu: z when z > 0 and z = 0 when z <= 0 
    """
    return np.maximum(z, 0)

def softmax(z : np): 
    """
        n is number of classes 
        m is number of training examples 
        z: shape(n, m)
    """ 
    z_stabilized = z - np.max(z, axis = 0, keepdims=True)
    return np.exp(z_stabilized) / (np.sum(np.exp(z_stabilized), axis = 0, keepdims=True))
def cce_loss(a3 : np, y : np): 
    """
        a3: (n3, m) (probability)
        y: (n3, m) (either 0 or 1)
        ccp_loss = -(1/m) (sum of training example) (sum of classes) yilog(ai)
        Return: 
            scalar 
    """
    m = a3.shape[1]
    per_class = np.sum(y * np.log(a3), axis=0, keepdims=True)
    return (-1/m) * np.sum(per_class)


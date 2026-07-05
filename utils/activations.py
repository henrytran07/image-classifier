import numpy as np


def z_score_normalization(x: np.ndarray, epsilon=1e-8):
    """
        Global (whole-array) standardization. Used for INPUT normalization only.
        NOTE: this is NOT batch normalization — it collapses the entire array to a
        single scalar mean/std. Batch norm (per-feature, over the batch axis) now
        lives inside layers.batch_norm_forward, so this function is no longer used
        in the network's hidden layers.
    """
    return (x - x.mean()) / np.sqrt(x.var() + epsilon)


def relu(z: np.ndarray):
    return np.maximum(z, 0)


def softmax(z: np.ndarray):
    """
        z: (n_classes, m)
    """
    z_stabilized = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z_stabilized)
    return e / np.sum(e, axis=0, keepdims=True)


def cce_loss(a4: np.ndarray, y: np.ndarray, epsilon=1e-12):
    """
        a4: (n_classes, m) softmax probabilities
        y:  (n_classes, m) one-hot
        Return: scalar categorical cross-entropy.
        a4 is clipped away from 0 to avoid log(0) -> -inf / nan.
    """
    m = a4.shape[1]
    a4 = np.clip(a4, epsilon, 1.0)
    per_example = np.sum(y * np.log(a4), axis=0, keepdims=True)
    return (-1 / m) * np.sum(per_example)

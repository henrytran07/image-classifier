from datasets import load_dataset
import numpy as np


def shuffle_data(x: np.ndarray, y: np.ndarray):
    """
        x: (784, m)
        y: (n, m)
        Return: shuffled_x, shuffled_y with the same shapes (columns permuted together)
    """
    m = x.shape[1]
    order = np.random.permutation(m)
    return x[:, order], y[:, order]


def _to_array(images):
    """Robustly convert a list of PIL images to a (m, 28, 28) uint8 array."""
    return np.stack([np.asarray(im) for im in images])


def load_mnist_ds():
    """
        Return:
            x_train: (m_train, 28, 28), y_train: (1, m_train)
            x_val:   (m_val, 28, 28),   y_val:   (1, m_val)
            x_test:  (m_test, 28, 28),  y_test:  (1, m_test)
    """
    ds = load_dataset("zalando-datasets/fashion_mnist")

    x = _to_array(ds['train']['image'])
    y = np.array(ds['train']['label']).reshape(1, -1)
    m = x.shape[0]

    order = np.random.permutation(m)
    x, y = x[order], y[:, order]

    split = int(0.8 * m)
    x_train, y_train = x[:split], y[:, :split]
    x_val, y_val = x[split:], y[:, split:]

    x_test = _to_array(ds['test']['image'])
    y_test = np.array(ds['test']['label']).reshape(1, -1)

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_dense_model_ds():
    """
        Flatten to (784, m) and standardize using TRAIN statistics only
        (so val/test see no information from their own distribution).
        Return:
            x_train_dm, y_train, x_val_dm, y_val, x_test_dm, y_test
    """
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_ds()

    x_train_f = x_train.reshape(x_train.shape[0], -1).T.astype(np.float64)  # (784, m)
    x_val_f = x_val.reshape(x_val.shape[0], -1).T.astype(np.float64)
    x_test_f = x_test.reshape(x_test.shape[0], -1).T.astype(np.float64)

    mean = x_train_f.mean()
    std = x_train_f.std() + 1e-8

    x_train_dm = (x_train_f - mean) / std
    x_val_dm = (x_val_f - mean) / std
    x_test_dm = (x_test_f - mean) / std

    return x_train_dm, y_train, x_val_dm, y_val, x_test_dm, y_test

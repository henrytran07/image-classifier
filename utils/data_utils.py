from datasets import load_dataset
import numpy as np 
from .activations import z_score_normalization

def load_mnist_ds(): 
    """
        fashion_mnist_ds
            dict(): 
                {'train': ['image', 'label'], 'test': ['image', 'label']}
        return: 
            x_train shape: (60000, 28, 28)
            y_train shape: (60000, 1)
            x_test shape: (10000, 28, 28)
            y_test shape: (10000, 1) 
    """
    fashion_mnist_ds = load_dataset("zalando-datasets/fashion_mnist")

    x_train = np.array(fashion_mnist_ds['train']['image'])
    y_train = np.array(fashion_mnist_ds['train']['label'])
    m_train = x_train.shape[0] if x_train.shape[0] == y_train.shape[0] else 0 
    y_train = y_train.reshape(m_train, -1)

    x_test = np.array(fashion_mnist_ds['test']['image'])
    y_test = np.array(fashion_mnist_ds['test']['label']) 
    m_test = x_test.shape[0] if x_test.shape[0] == y_test.shape[0] else 0 
    y_test = y_test.reshape(m_test, -1)
    return x_train, y_train, m_train, x_test, y_test, m_test

def load_dense_model_ds(): 
    """
        Return: 
            x_train_dm_normalized shape: (784, m_train)
            y_train shape: (60000, 1)
            x_test_dm_normalized shape: (784, m_test)
            y_test shape: (10000, 1) 
    """
    x_train, y_train, m_train, x_test, y_test, m_test = load_mnist_ds()
    x_train_dm_normalized = z_score_normalization(x_train.reshape(-1, m_train))
    x_test_dm_normalized = z_score_normalization(x_test.reshape(-1, m_test))
    return x_train_dm_normalized, y_train, x_test_dm_normalized, y_test 

def load_cnn_model_ds(): 
    """
        Return: 
            x_train_cnn shape: (60000, 28, 28, 3)
            y_train shape: (60000, 1)
            x_test_cnn shape: (10000, 28, 28, 3)
            y_test shape: (10000, 1) 
    """
    x_train, y_train, _, x_test, y_test, _ = load_mnist_ds()
    x_train_sliced_norm = z_score_normalization(x_train[:])
    x_train_cnn = np.stack([x_train_sliced_norm, x_train_sliced_norm, x_train_sliced_norm], axis=3)

    x_test_sliced_norm = x_test[:]
    x_test_cnn = np.stack([x_test_sliced_norm, x_test_sliced_norm, x_test_sliced_norm], axis=3)
    return x_train_cnn, y_train, x_test_cnn, y_test 

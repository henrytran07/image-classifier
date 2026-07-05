from .layers import *
from .activations import cce_loss
from .data_utils import shuffle_data
import numpy as np


class dense_model:
    def __init__(self, learning_rate=1e-3, epochs=20, mini_batch_size=64,
                 keep_prob=0.8, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 bn_momentum=0.9):
        self.linear_params = init_linear_params()
        self.bn_params = init_bn_params()
        self.bn_running = init_bn_running_stats()
        self.adam_state = init_adam_state(self.bn_params, self.linear_params)
        self.alpha = learning_rate
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.keep_prob = keep_prob
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.bn_momentum = bn_momentum
        self.t = 1

    def _one_hot(self, y, n_classes=10):
        m = y.shape[1]
        oh = np.zeros((n_classes, m))
        oh[y.flatten(), np.arange(m)] = 1
        return oh

    def fit(self, x_train, y_train, x_val, y_val):
        y_train_oh = self._one_hot(y_train)
        y_val_oh = self._one_hot(y_val)
        for epoch in range(self.epochs):
            sx, sy = shuffle_data(x_train, y_train_oh)
            mini_batches = get_mini_batches(sx, sy, self.mini_batch_size)
            epoch_loss = 0.0
            for mbx, mby in mini_batches:
                cache, A4 = forward_prop(self.linear_params, self.bn_params, mbx,
                                         training=True, keep_prob=self.keep_prob,
                                         bn_running=self.bn_running,
                                         momentum=self.bn_momentum)
                epoch_loss += cce_loss(A4, mby)
                grads = backward_prop(A4, mbx, mby, cache, self.linear_params,
                                      self.keep_prob)
                self.adam_state, corrected = adam_optimizer(
                    grads, self.adam_state, self.t, self.beta1, self.beta2)
                self.linear_params, self.bn_params = update_adam(
                    self.linear_params, self.bn_params, corrected,
                    self.alpha, self.epsilon)
                self.t += 1

            _, A4_val = forward_prop(self.linear_params, self.bn_params, x_val,
                                     training=False, bn_running=self.bn_running)
            val_loss = cce_loss(A4_val, y_val_oh)
            yhat_val = np.argmax(A4_val, axis=0, keepdims=True)
            val_acc = float(np.mean(yhat_val == y_val))
            print(f"Epoch {epoch+1}/{self.epochs} — "
                  f"train_loss: {epoch_loss/len(mini_batches):.4f} — "
                  f"val_loss: {val_loss:.4f} — val_accuracy: {val_acc:.4f}")

    def predict(self, X):
        _, A4 = forward_prop(self.linear_params, self.bn_params, X,
                             training=False, bn_running=self.bn_running)
        return np.argmax(A4, axis=0, keepdims=True)

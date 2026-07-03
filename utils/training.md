## Model Architecture and Formulas

### Architecture

| Layer | Activation | Output Shape |
|-------|------------|--------------|
| Input | —          | (784, m)     |
| 1     | ReLU       | (512, m)     |
| 2     | ReLU       | (256, m)     |
| 3     | ReLU       | (128, m)     |
| 4     | Softmax    | (10, m)      |

### Softmax

$$\sigma(z_i) = a_i = \frac{e^{z_i}}{\sum_{j=1}^{c} e^{z_j}}$$

### Categorical cross-entropy loss

$$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{c} y_{ij} \cdot \log(a_{ij})$$

---

## Backpropagation Derivation

### Linear step (layer 4)

$$z_k = W_4[k, :] \cdot A_3 + b_4$$

<img src="/images/zk_formula.svg" alt="z_k illustration" height="300">

### Chain rule

$$\frac{\partial L}{\partial W_4} = \frac{\partial L}{\partial a_4} \cdot \frac{\partial a_i}{\partial z_k} \cdot \frac{\partial z_k}{\partial W_4}$$

### Softmax Jacobian

$$\frac{\partial a_i}{\partial z_k} = \begin{cases} a_i(1 - a_i) & \text{if } i = k \\ -a_i \, a_k & \text{if } i \neq k \end{cases}$$

### Loss gradient w.r.t. $a_k$

$$\frac{\partial L}{\partial a_k} = -\frac{y_k}{a_k} \quad \text{where } k \text{ is the class index}$$

### Combined gradient (softmax + cross-entropy)

$$\frac{\partial L}{\partial z_k} = a_k - y_k$$

### Matrix form

$$\frac{\partial L}{\partial Z_4} = A_4 - Y \quad \in \mathbb{R}^{10 \times m}$$

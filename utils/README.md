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

$$\frac{\partial L}{\partial b_4} = np.sum(dZ_4, axis=1, keepdims = True) \quad \in \mathbb{R}^{10 \times 1}$$


$$\frac{\partial L}{\partial z_3} = W_4^T \cdot (A_4 - Y) \odot f'(z_3) \quad \in \mathbb{R}^{128 \times m}$$

$$\frac{\partial L}{\partial W_3} = dZ_3 \cdot A_2^T \quad \in \mathbb{R} ^{128 \times 256}$$

$$\frac{\partial L}{\partial b_3} = np.sum(dZ_3, axis=1) \quad \in \mathbb{R}^{128 \times 1}$$

$$\frac{\partial L}{\partial z_2} = W_3^T \cdot dZ_3\odot f'(z_2) \quad \in \mathbb{R}^{256 \times m}$$

$$\frac{\partial L}{\partial W_2} = dZ_2 \cdot A_1^T \quad \in \mathbb{R}^{256 \times 512}$$

$$\frac{\partial L}{\partial b_2} = \text{np.sum}(dZ_2, \text{axis}=1, \text{keepdims}=True) \quad \in \mathbb{R}^{256 \times 1}$$

$$\frac{\partial L}{\partial z_1} = W_2^T \cdot dZ_2 \odot f'(z_1) \quad \in \mathbb{R}^{512 \times m}$$

$$\frac{\partial L}{\partial W_1} = dZ_1 \cdot X^T \quad \in \mathbb{R}^{512 \times 784}$$

$$\frac{\partial L}{\partial b_1} = \text{np.sum}(dZ_1, \text{axis}=1, \text{keepdims}=True) \quad \in \mathbb{R}^{512 \times 1}$$
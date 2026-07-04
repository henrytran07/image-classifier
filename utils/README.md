## Model Architecture and Formulas

### Architecture

| Layer | Activation | Output Shape | Learnable Params ($\gamma$ and $\beta$) |
|-------|------------|--------------|----------------------------------------|
| Input | —          | (784, m)     | —                                      |
| 1     | ReLU       | (512, m)     | (512, 1) and (512, 1)                  |
| 2     | ReLU       | (256, m)     | (256, 1) and (256, 1)                  |
| 3     | ReLU       | (128, m)     | (128, 1) and (128, 1)                  |
| 4     | Softmax    | (10, m)      | —                                      |
### Softmax

$$\sigma(z_i) = a_i = \frac{e^{z_i}}{\sum_{j=1}^{c} e^{z_j}}$$

### Batch Normalization
m is batch size

- Find batch mean:
$$\mu = \frac{1}{m}\sum_{i=1}^{m} z_i$$

- Find batch variance:
$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(z_i - \mu)^2$$

- Normalize:
$$\hat{z} = \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

- Scale and shift:
$$z = \gamma \cdot \hat{z} + \beta$$

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

$$\frac{\partial L}{\partial W_4} = \frac{1}{m} dZ_4 \cdot A_3^T \quad \in \mathbb{R} ^{10 \times 128}$$

$$\frac{\partial L}{\partial b_4} = \frac{1}{m} \text{np.sum}(dZ_4, axis=1, keepdims = True) \quad \in \mathbb{R}^{10 \times 1}$$


$$\frac{\partial L}{\partial z_3} = W_4^T \cdot (A_4 - Y) \odot f'(z_3) \quad \in \mathbb{R}^{128 \times m}$$

$$\frac{\partial L}{\partial W_3} = \frac{1}{m} dZ_3 \cdot A_2^T \quad \in \mathbb{R} ^{128 \times 256}$$

$$\frac{\partial L}{\partial b_3} = \frac{1}{m} \text{np.sum}(dZ_3, axis=1) \quad \in \mathbb{R}^{128 \times 1}$$

$$\frac{\partial L}{\partial z_2} = W_3^T \cdot dZ_3\odot f'(z_2) \quad \in \mathbb{R}^{256 \times m}$$

$$\frac{\partial L}{\partial W_2} = \frac{1}{m} dZ_2 \cdot A_1^T \quad \in \mathbb{R}^{256 \times 512}$$

$$\frac{\partial L}{\partial b_2} = \frac{1}{m} \text{np.sum}(dZ_2, \text{axis}=1, \text{keepdims}=True) \quad \in \mathbb{R}^{256 \times 1}$$

$$\frac{\partial L}{\partial z_1} = W_2^T \cdot dZ_2 \odot f'(z_1) \quad \in \mathbb{R}^{512 \times m}$$

$$\frac{\partial L}{\partial W_1} = \frac{1}{m}  dZ_1 \cdot X^T \quad \in \mathbb{R}^{512 \times 784}$$

$$\frac{\partial L}{\partial b_1} = \frac{1}{m} \text{np.sum}(dZ_1, \text{axis}=1, \text{keepdims}=True) \quad \in \mathbb{R}^{512 \times 1}$$

$$\frac{\partial L}{\partial \gamma} = \text{np.sum}(dZ \odot \hat{z}, \text{axis}=1, \text{keepdims}=True)$$

$$\frac{\partial L}{\partial \beta} = \text{np.sum}(dZ, \text{axis}=1, \text{keepdims}=True)$$

### Adam Optimizer 
$v_{dW} = \beta_1v_{dW} + (1 - \beta_1)dW$
$s_{dW} = \beta_2s_{dW} + (1 - \beta_2)dW$

$v_{db} = \beta_1v_{db} + (1 - \beta_1)db$
$s_{db} = \beta_2s_{db} + (1 - \beta_2)db$

$v_{d\gamma} = \beta_1v_{d\gamma} + (1 - \beta_1)d\gamma$
$s_{d\gamma} = \beta_2s_{d\gamma} + (1 - \beta_2)d\gamma$

$v_{d\beta} = \beta_1v_{d\beta} + (1 - \beta_1)d\beta$
$s_{d\beta} = \beta_2s_{d\beta} + (1 - \beta_2)d\beta$
### Bias Correction 

$v_{dW}^{corr} = \frac{v_{dW}}{1 - \beta_1^t} \quad \quad s_{dW}^{corr} = \frac{s_{dW}}{1 - \beta_2^t}$ with t is number of steps 

$v_{db}^{corr} = \frac{v_{db}}{1 - \beta_1^t} \quad \quad s_{db}^{corr} = \frac{s_{db}}{1 - \beta_2^t}$ 

$v_{d\gamma}^{corr} = \frac{v_{d\gamma}}{1 - \beta_1^t} \quad \quad s_{d\gamma}^{corr} = \frac{s_{d\gamma}}{1 - \beta_2^t}$ 

$v_{d\beta}^{corr} = \frac{v_{d\beta}}{1 - \beta_1^t} \quad \quad s_{d\beta}^{corr} = \frac{s_{d\beta}}{1 - \beta_2^t}$ 


### Update 
$W = W - \alpha \frac{v_{dW}^{corr}}{\sqrt{s_{dW}^{corr} + \epsilon}} \quad \quad b = b - \alpha \frac{v_{db}^{corr}}{\sqrt{s_{db}^{corr} + \epsilon}}$

$\gamma = \gamma - \alpha \frac{v_{d\gamma}^{corr}}{\sqrt{s_{d\gamma}^{corr} + \epsilon}} \quad \quad \beta = \beta - \alpha \frac{v_{d\beta}^{corr}}{\sqrt{s_{d\beta}^{corr} + \epsilon}}$


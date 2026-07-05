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

`u = W @ A_prev + b` is the pre-activation for a hidden layer, shape `(n, m)`.
Statistics are computed **per feature** (one mean/variance per row), across the
batch axis — *not* one global scalar over the whole matrix.

- Batch mean per feature:
$$\mu = \frac{1}{m}\sum_{i=1}^{m} u_i \quad \in \mathbb{R}^{n \times 1}$$

- Batch variance per feature:
$$\sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(u_i - \mu)^2 \quad \in \mathbb{R}^{n \times 1}$$

- Normalize:
$$\hat{z} = \frac{u - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

- Scale and shift:
$$z = \gamma \odot \hat{z} + \beta$$

**Running statistics for inference.** At eval time there is no batch to compute
`μ`/`σ²` from, so each BN layer keeps an exponential moving average of the
training-time batch statistics:

$$\text{running\_mean} \leftarrow \text{momentum} \cdot \text{running\_mean} + (1-\text{momentum}) \cdot \mu$$
$$\text{running\_var} \leftarrow \text{momentum} \cdot \text{running\_var} + (1-\text{momentum}) \cdot \sigma^2$$

with `momentum = 0.9`. Inference normalizes with `running_mean`/`running_var`
instead of a fresh batch `μ`/`σ²`.

**Why the bias `b` before BN is inert.** Because BN immediately subtracts the
batch mean, any constant bias added to `u` cancels out: `db1`, `db2`, `db3`
are always ~0. `beta` is the effective bias for layers 1–3; `b1`–`b3` are kept
only to mirror the architecture table above and could be dropped without
changing the model.

### Dropout Regularization

- `keep_prob = 0.8` — during each forward pass, every neuron in a hidden layer has an 80% chance of being active and a 20% chance of being temporarily dropped (set to zero). This forces the network to learn redundant representations and prevents over-reliance on any single neuron, reducing overfitting.

### Categorical cross-entropy loss

$$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{c} y_{ij} \cdot \log(a_{ij})$$

`a` is clipped to `[\epsilon, 1]` (`\epsilon = 10^{-12}`) before the log to avoid
`-inf`/`nan` if softmax underflows a probability to exactly 0.

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

### Matrix form — output layer (no BN here, so this part is unchanged)

$$\frac{\partial L}{\partial Z_4} = A_4 - Y \quad \in \mathbb{R}^{10 \times m}$$

$$\frac{\partial L}{\partial W_4} = \frac{1}{m} dZ_4 \cdot A_3^T \quad \in \mathbb{R} ^{10 \times 128}$$

$$\frac{\partial L}{\partial b_4} = \frac{1}{m} \text{np.sum}(dZ_4, axis=1, keepdims = True) \quad \in \mathbb{R}^{10 \times 1}$$

### Hidden layers (through BN) — layer $l \in \{1,2,3\}$

Let `dZ_l` be the gradient arriving at the batch-normalized activation `z_l`
(after passing back through dropout's mask and ReLU's derivative, same as
before). Unlike a plain linear layer, `dZ_l` does **not** flow straight into
`du_l = W_l @ A_{prev} + b_l`; it first has to go back through the
normalization step, because `μ` and `σ²` are themselves functions of every
example in the batch.

$$d\gamma_l = \frac{1}{m}\sum dZ_l \odot \hat{z}_l \quad \in \mathbb{R}^{n_l \times 1}$$

$$d\beta_l = \frac{1}{m}\sum dZ_l \quad \in \mathbb{R}^{n_l \times 1}$$

$$g = dZ_l \odot \gamma_l \quad \text{(gradient w.r.t. } \hat{z}_l\text{)}$$

$$\frac{\partial L}{\partial u_l} = \frac{1}{\sqrt{\sigma_l^2+\epsilon}}\left(g - \overline{g} - \hat{z}_l \odot \overline{g \odot \hat{z}_l}\right)$$

where $\overline{(\cdot)}$ denotes the mean over the batch axis (`axis=1`).
This is the standard batch-norm backward formula — the extra two correction
terms exist precisely because `μ` and `σ²` depend on the whole batch, so a
perturbation to any single example nudges every other example's normalized
value too.

Once `du_l = ∂L/∂u_l` is known, the linear-layer gradients follow exactly as
before:

$$\frac{\partial L}{\partial W_l} = \frac{1}{m}\, du_l \cdot A_{prev}^T$$

$$\frac{\partial L}{\partial b_l} = \frac{1}{m}\text{np.sum}(du_l, \text{axis}=1, \text{keepdims}=True) \approx 0 \text{ (see note above)}$$

$$\frac{\partial L}{\partial A_{prev}} = W_l^T \cdot du_l$$

which is what continues backward into the next (earlier) layer's dropout mask
and ReLU derivative.

> **Previous version of this doc / earlier code treated BN as pass-through
> during backprop** (i.e. used `dZ` directly as `du`, and computed `dgamma`
> against `z` instead of `\hat{z}`). That undercounts the batch-coupling terms
> above and gives gradients for `W1–W3`, `b1–b3`, and `gamma1–gamma3` that are
> off by 10–30% relative to the true gradient (verified via numerical
> gradient checking). The formulas above reflect the corrected implementation.

### Adam Optimizer
$$v_{dW} = \beta_1v_{dW} + (1 - \beta_1)dW$$
$$s_{dW} = \beta_2s_{dW} + (1 - \beta_2)dW^2$$

$$v_{db} = \beta_1v_{db} + (1 - \beta_1)db$$
$$s_{db} = \beta_2s_{db} + (1 - \beta_2)db^2$$

$$v_{d\gamma} = \beta_1v_{d\gamma} + (1 - \beta_1)d\gamma$$
$$s_{d\gamma} = \beta_2s_{d\gamma} + (1 - \beta_2)d\gamma^2$$

$$v_{d\beta} = \beta_1v_{d\beta} + (1 - \beta_1)d\beta$$
$$s_{d\beta} = \beta_2s_{d\beta} + (1 - \beta_2)d\beta^2$$

### Bias Correction

$v_{dW}^{corr} = \frac{v_{dW}}{1 - \beta_1^t} \quad \quad s_{dW}^{corr} = \frac{s_{dW}}{1 - \beta_2^t}$ with t is number of steps

$v_{db}^{corr} = \frac{v_{db}}{1 - \beta_1^t} \quad \quad s_{db}^{corr} = \frac{s_{db}}{1 - \beta_2^t}$

$v_{d\gamma}^{corr} = \frac{v_{d\gamma}}{1 - \beta_1^t} \quad \quad s_{d\gamma}^{corr} = \frac{s_{d\gamma}}{1 - \beta_2^t}$

$v_{d\beta}^{corr} = \frac{v_{d\beta}}{1 - \beta_1^t} \quad \quad s_{d\beta}^{corr} = \frac{s_{d\beta}}{1 - \beta_2^t}$

### Update

$W = W - \alpha \dfrac{v_{dW}^{corr}}{\sqrt{s_{dW}^{corr}} + \epsilon} \quad \quad b = b - \alpha \dfrac{v_{db}^{corr}}{\sqrt{s_{db}^{corr}} + \epsilon}$

$\gamma = \gamma - \alpha \dfrac{v_{d\gamma}^{corr}}{\sqrt{s_{d\gamma}^{corr}} + \epsilon} \quad \quad \beta = \beta - \alpha \dfrac{v_{d\beta}^{corr}}{\sqrt{s_{d\beta}^{corr}} + \epsilon}$

> Note the `+ \epsilon` sits **outside** the square root (`\sqrt{s^{corr}} + \epsilon`),
> matching the code — the earlier version of this doc showed `\sqrt{s^{corr} + \epsilon}`
> (epsilon inside the root), which is a common alternate convention but not
> what's implemented here.

### Split Training Examples into Mini-Batches

- Batch size: 64

**Shuffle** training examples at the start of each epoch:
```python
order = np.random.permutation(m)
X = X[:, order]
Y = Y[:, order]
```

**Partition** into batches:
- `num_batches = m // batch_size`
- `remainder = m % batch_size`

For each batch `i` in `range(num_batches)`:
- `start = i * batch_size`
- `end = (i + 1) * batch_size`
- `X_batch = X[:, start:end]` — shape `(784, 64)`
- `Y_batch = Y[:, start:end]` — shape `(10, 64)`

If `remainder > 0`, the last batch covers `X[:, num_batches * batch_size:]`

---

## Implementation Notes / Changelog

- **Batch normalization fixed.** Forward and backward pass now match the
  standard per-feature BN formulation described above (previously, the "batch
  norm" was actually a single global mean/std over the whole array, and the
  backward pass didn't propagate gradient through the normalization step at
  all). Verified against numerical gradients for every layer
  (`W1–W4, b4, gamma1–3, beta1–3`) at ~1e-10 relative error.
- **Running mean/var added** so inference (`training=False`) no longer
  normalizes using whatever batch happens to be passed in.
- **`cce_loss` clips probabilities** away from 0 before taking `log`, avoiding
  occasional `nan` losses.
- **`data_utils`**: validation/test sets are now standardized using the
  *training* set's mean/std, rather than each split's own statistics.

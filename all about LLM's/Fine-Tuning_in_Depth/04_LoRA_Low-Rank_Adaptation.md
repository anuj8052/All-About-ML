# Module 4: LoRA (Low-Rank Adaptation)

## 1. The Core Idea Behind LoRA

LoRA is one of the most popular and effective PEFT techniques. It is based on a fascinating hypothesis: even though a pre-trained model's weight matrices have a full rank (meaning they contain a rich and complex set of information), the **change** in those weights during model adaptation (the "weight update" matrix) has a very low "intrinsic rank".

In simpler terms, you don't need to modify all the weights in a large matrix to teach the model a new task. You can achieve similar performance by making a small, carefully structured change that can be represented much more efficiently. LoRA proposes that this efficient change can be captured by a low-rank matrix.

## 2. The Mathematics of LoRA

To understand LoRA, let's first recall the update in Full Fine-Tuning (FFT). For a given weight matrix $W_0 \in \mathbb{R}^{d \times k}$, FFT learns an update matrix $\Delta W \in \mathbb{R}^{d \times k}$ such that the new weight matrix is $W = W_0 + \Delta W$. Both $W_0$ and $\Delta W$ are large, dense matrices.

LoRA's key insight is to **approximate** the update matrix $\Delta W$ using two much smaller, low-rank matrices. This is called a **low-rank decomposition**.

$$ \Delta W \approx B \cdot A $$

Where:
-   $A \in \mathbb{R}^{r \times k}$ is a matrix with a very small row dimension, $r$.
-   $B \in \mathbb{R}^{d \times r}$ is a matrix with a very small column dimension, $r$.
-   **$r$ is the rank of the decomposition**, and it is a crucial hyperparameter. Typically, $r \ll d$ and $r \ll k$. For example, $d$ and $k$ could be 4096, while $r$ might be just 8 or 16.

### How it Works in Practice

1.  **Initialization:** The weights of the pre-trained model, $W_0$, are **frozen**. They do not receive any updates during training. The matrix $A$ is initialized with small random values (e.g., from a Gaussian distribution), and the matrix $B$ is initialized with zeros. This ensures that at the beginning of training, $\Delta W = B \cdot A$ is zero, and the model's performance is identical to the pre-trained base model.

2.  **Training:** During fine-tuning, we only learn the weights of matrices $A$ and $B$. The number of trainable parameters is the sum of the sizes of $A$ and $B$, which is $(r \times k) + (d \times r)$. This is vastly smaller than the $(d \times k)$ parameters in the original matrix $W_0$.

3.  **Modified Forward Pass:** The forward pass of a LoRA-adapted layer is modified. Instead of just computing $h = W_0 x$, we compute:
    $$ h = W_0 x + (B A) x = W_0 x + B (A x) $$
    The calculation is often scaled by a constant, `alpha`, which is another hyperparameter. A common practice is to set `alpha` equal to the rank `r`. The final equation becomes:
    $$ h = W_0 x + \frac{\alpha}{r} (B A) x $$
    This scaling helps to stabilize training.

### Example: Parameter Savings

Imagine a weight matrix in a transformer where $d=4096$ and $k=4096$.
-   Number of parameters in $W_0$: $4096 \times 4096 = 16,777,216$.
-   Now, let's use LoRA with a rank $r=8$.
    -   Number of parameters in $A$ ($r \times k$): $8 \times 4096 = 32,768$.
    -   Number of parameters in $B$ ($d \times r$): $4096 \times 8 = 32,768$.
    -   Total trainable parameters for this layer: $32,768 + 32,768 = 65,536$.

This is a **~256x reduction** in the number of trainable parameters for this single layer!

## 3. Deployment

Once training is complete, for maximum efficiency during inference (so you don't have to compute the two matrix multiplications separately), you can merge the learned LoRA weights back into the base model.
You simply compute the final weight matrix $W = W_0 + B A$ and save this new, full-sized matrix. This results in no inference latency compared to the original model.

## 4. Key Hyperparameters

-   `r`: The rank of the decomposition. This is the most important hyperparameter. A higher `r` means more trainable parameters and more expressive power, but also higher training cost. Common values are 8, 16, 32, 64.
-   `lora_alpha`: The scaling factor. It's often set to be the same as `r`, but can be tuned. Some libraries set it to $2 \times r$ by default.
-   `target_modules`: A list of which layers in the transformer to apply LoRA to. It's most commonly applied to the attention mechanism's query and value projection matrices (`q_proj` and `v_proj`).

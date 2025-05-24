# 03. Activation Functions in Neural Networks

Activation functions are a critical component of neural networks. They determine the output of a neuron given a set of inputs and are responsible for introducing non-linearity into the network, which is essential for learning complex patterns.

## Purpose of Activation Functions: Introducing Non-linearity

If a neural network were to only use linear transformations (like the weighted sum of inputs), the entire network would behave like a single linear model, regardless of how many layers it has.
`Output = w_n * (... (w_2 * (w_1 * x + b_1) + b_2) ... ) + b_n`
This expression can be simplified into `Output = W_final * x + B_final`.

A linear model has limited representational power. It can only learn linear relationships between inputs and outputs. However, most real-world data and problems are non-linear.

Activation functions introduce non-linearity by transforming the weighted sum of inputs at each neuron. This allows neural networks to:
*   Approximate complex, non-linear functions.
*   Learn intricate patterns in the data.
*   Create hierarchical feature representations in deep networks.

Without non-linear activation functions, a deep neural network would effectively collapse into a single-layer linear network.

## Sigmoid Function

The Sigmoid function, also known as the logistic function, was historically popular, especially for binary classification problems in the output layer.

*   **Formula:**
    `σ(z) = 1 / (1 + e^(-z))`
    where `z` is the weighted sum of inputs plus bias (`w · x + b`).

*   **Graph:**
    *   Shape: S-shaped curve.
    *   Output Range: (0, 1).
    *   As `z` approaches -∞, `σ(z)` approaches 0.
    *   As `z` approaches +∞, `σ(z)` approaches 1.
    *   `σ(0) = 0.5`.
    ```
    Graph Description:
    Y-axis: σ(z) (from 0 to 1)
    X-axis: z (input)
    The curve starts near 0 for large negative z, smoothly rises, passes through (0, 0.5), and flattens out near 1 for large positive z.
    ```

*   **Pros:**
    *   **Probabilistic Interpretation:** Output between 0 and 1 can be interpreted as a probability, which is useful for binary classification output layers.
    *   **Smooth Gradient:** The function is differentiable everywhere, providing a smooth gradient for optimization.

*   **Cons:**
    *   **Vanishing Gradient Problem:** For very high or very low values of `z`, the gradient of the sigmoid function becomes very small (close to zero). During backpropagation, these small gradients are multiplied through layers, leading to vanishingly small updates for weights in the earlier layers. This significantly slows down or even halts the learning process for deep networks.
        *   The derivative `σ'(z) = σ(z) * (1 - σ(z))` has a maximum value of 0.25 at `z=0`. When `z` is large (positive or negative), `σ'(z)` is close to 0.
    *   **Not Zero-Centered Output:** The output is always positive (between 0 and 1). This means the gradients for the weights feeding into a neuron with sigmoid activation will always have the same sign (either all positive or all negative, depending on the gradient of the loss function w.r.t the neuron's output). This can lead to inefficient, zig-zagging updates during gradient descent.
    *   **Computationally Expensive:** The exponential function `e^(-z)` can be computationally intensive compared to simpler functions like ReLU.

## Tanh Function (Hyperbolic Tangent)

The Tanh function is another S-shaped function, similar to the sigmoid but with an output range of (-1, 1).

*   **Formula:**
    `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
    It can also be expressed in terms of the sigmoid function:
    `tanh(z) = 2 * σ(2z) - 1`

*   **Graph:**
    *   Shape: S-shaped curve, similar to sigmoid but centered at 0.
    *   Output Range: (-1, 1).
    *   As `z` approaches -∞, `tanh(z)` approaches -1.
    *   As `z` approaches +∞, `tanh(z)` approaches 1.
    *   `tanh(0) = 0`.
    ```
    Graph Description:
    Y-axis: tanh(z) (from -1 to 1)
    X-axis: z (input)
    The curve starts near -1 for large negative z, smoothly rises, passes through (0, 0), and flattens out near 1 for large positive z.
    ```

*   **Pros:**
    *   **Zero-Centered Output:** The output range is (-1, 1), which is zero-centered. This helps to make the learning process more efficient as gradients are more likely to be balanced around zero, mitigating the issue seen with sigmoid where gradients for weights are always of the same sign. This often leads to faster convergence than sigmoid.
    *   **Smooth Gradient:** Like sigmoid, it's differentiable everywhere.

*   **Cons:**
    *   **Vanishing Gradient Problem:** Still suffers from the vanishing gradient problem for very high or very low values of `z` (though the derivative `tanh'(z) = 1 - tanh^2(z)` has a range of (0, 1], which is steeper than sigmoid's derivative).
    *   **Computationally Expensive:** Involves exponential functions, similar to sigmoid.

## ReLU (Rectified Linear Unit)

ReLU is currently one of the most widely used activation functions in deep learning, especially in hidden layers.

*   **Formula:**
    `ReLU(z) = max(0, z)`
    This means:
    *   If `z > 0`, `ReLU(z) = z`
    *   If `z <= 0`, `ReLU(z) = 0`

*   **Graph:**
    *   Shape: Linear for positive values, zero for negative values. A sharp corner at `z=0`.
    *   Output Range: [0, ∞).
    ```
    Graph Description:
    Y-axis: ReLU(z) (from 0 to positive values)
    X-axis: z (input)
    For z < 0, the graph is a horizontal line at Y=0.
    At z = 0, the graph abruptly changes direction.
    For z > 0, the graph is a straight line with a slope of 1 (Y=z).
    ```

*   **Pros:**
    *   **Computational Efficiency:** Very simple to compute (a single comparison and possibly an assignment), making it much faster than sigmoid or tanh.
    *   **Mitigates Vanishing Gradient (in positive region):** For positive inputs (`z > 0`), the derivative is constant (1). This means that for active neurons, the gradient can flow through the network without diminishing, which helps with training deeper networks.
    *   **Sparsity:** For negative inputs, the output is zero. This means that some neurons become inactive (outputting zero), leading to sparse representations in the network. Sparsity can be beneficial for generalization and can make the network more robust.

*   **Cons:**
    *   **Dying ReLU Problem:** If a neuron's input `z` consistently becomes negative during training, it will output 0. The gradient for `z < 0` is also 0. Consequently, the weights of this neuron will not be updated anymore, and the neuron effectively "dies" – it stops learning. This can happen if the learning rate is too high or if there's a large negative bias.
    *   **Not Zero-Centered Output:** Similar to sigmoid, the output is always non-negative (0 or positive). This can lead to the same issues with gradient updates as sigmoid, though typically less severe due to other benefits.
    *   **Non-Differentiable at z=0:** The function has a sharp corner at `z=0` and is not differentiable there. However, in practice, this is usually not a major issue. A sub-gradient (0 or 1) is typically used at `z=0`.

## Leaky ReLU

Leaky ReLU is an attempt to address the "Dying ReLU" problem.

*   **Formula:**
    `LeakyReLU(z) = max(αz, z)`
    This means:
    *   If `z > 0`, `LeakyReLU(z) = z`
    *   If `z <= 0`, `LeakyReLU(z) = αz`
    where `α` (alpha) is a small positive constant, typically around 0.01.

*   **Graph:**
    *   Shape: Similar to ReLU, but with a small non-zero slope for negative values.
    *   Output Range: (-∞, ∞) (though the negative part has a much smaller slope).
    ```
    Graph Description:
    Y-axis: LeakyReLU(z)
    X-axis: z (input)
    For z > 0, the graph is a straight line with a slope of 1 (Y=z).
    For z <= 0, the graph is a straight line with a small positive slope α (Y=αz).
    There's still a kink at z=0, but the line for negative z is not flat on the x-axis.
    ```

*   **Advantages over ReLU:**
    *   **Prevents Dying Neurons:** By allowing a small, non-zero gradient when the unit is not active (`z < 0`), Leaky ReLU enables updates to the weights even for neurons that predominantly receive negative input. This helps prevent them from dying.
    *   The derivative for `z < 0` is `α`, not 0.

*   **Cons:**
    *   The results are not always consistently better than ReLU.
    *   The value of `α` is another hyperparameter to tune.

## Parametric ReLU (PReLU)

PReLU is a variation of Leaky ReLU where the slope `α` for negative inputs is learned during training, rather than being a fixed hyperparameter.

*   **Formula:**
    `PReLU(z) = max(α_i z, z)`
    where `α_i` is a learnable parameter, specific to each neuron `i` (or shared across a layer).
*   **Brief Explanation:**
    *   The network learns the best value of `α` during backpropagation by updating it similarly to how weights are updated.
    *   If `α_i` is learned to be 0, PReLU becomes ReLU. If it's a small fixed value, it's like Leaky ReLU.
    *   Can potentially offer better performance by adapting the negative slope to the data.
    *   Adds more parameters to the model, increasing complexity.

## Exponential Linear Unit (ELU)

ELU is another activation function that aims to address the issues of ReLU and provide a zero-centered-like behavior for negative inputs.

*   **Formula:**
    *   If `z > 0`, `ELU(z) = z`
    *   If `z <= 0`, `ELU(z) = α(e^z - 1)`
    where `α` is a positive hyperparameter that controls the saturation point for negative inputs. Often `α=1`.

*   **Brief Explanation:**
    *   **Advantages:**
        *   For `z > 0`, it behaves like ReLU (no saturation, efficient computation).
        *   For `z < 0`, it smoothly saturates to `-α`. This can help push the mean activations closer to zero, similar to Tanh, which can speed up learning.
        *   Avoids the dying ReLU problem by having non-zero gradients for negative inputs.
    *   **Disadvantages:**
        *   More computationally expensive than ReLU due to the exponential function for negative inputs.
        *   Introduces the hyperparameter `α`.

## Softmax Function

The Softmax function is primarily used in the **output layer** of a neural network for **multi-class classification** problems. It converts a vector of raw scores (logits) into a probability distribution, where each element represents the probability of the input belonging to a particular class.

*   **Formula:**
    For a vector of `K` raw scores `z = (z_1, z_2, ..., z_K)`, the Softmax function computes the probability `p_j` for the `j`-th class as:
    `p_j = Softmax(z_j) = e^(z_j) / Σ_{k=1}^{K} e^(z_k)`
    for `j = 1, ..., K`.

*   **Properties:**
    *   **Output Range:** Each `p_j` is between 0 and 1.
    *   **Sum to One:** The sum of all probabilities `Σ p_j` equals 1.
    *   This makes the output interpretable as a probability distribution over the `K` classes.

*   **Use:**
    *   Typically used in conjunction with a cross-entropy loss function for training multi-class classifiers.
    *   The class with the highest probability is chosen as the network's prediction.
    *   Not typically used in hidden layers.

## Choosing the Right Activation Function

There's no single "best" activation function that works for all problems. The choice often depends on the specific task, network architecture, and empirical results. However, some general guidelines are:

1.  **ReLU as a Default for Hidden Layers:**
    *   Start with ReLU for hidden layers. It's computationally efficient and generally works well.
    *   Be mindful of the dying ReLU problem. If it occurs (many neurons stuck at zero output), consider alternatives.

2.  **Alternatives to ReLU (if needed):**
    *   **Leaky ReLU or PReLU:** If you suspect dying ReLUs are an issue, these are good alternatives. PReLU might offer slightly better performance at the cost of more parameters.
    *   **ELU:** Can sometimes provide better results than ReLUs, especially if negative values are meaningful and zero-centering is beneficial, but at a higher computational cost.
    *   **Maxout:** Another alternative that generalizes ReLU and Leaky ReLU. It outputs the maximum of several linear inputs. More computationally expensive.

3.  **Sigmoid and Tanh (Use with Caution in Hidden Layers):**
    *   Due to the vanishing gradient problem, Sigmoid and Tanh are generally not recommended for deep hidden layers.
    *   Tanh often performs better than Sigmoid in hidden layers if a saturating activation is needed, due to its zero-centered output.
    *   They can still be useful in specific architectures (e.g., RNNs sometimes use Tanh or Sigmoid for gating mechanisms).

4.  **Output Layer Activation Functions:**
    *   **Binary Classification:** Sigmoid function (to output a probability).
    *   **Multi-class Classification:** Softmax function (to output a probability distribution over classes).
    *   **Regression (predicting continuous values):** Linear activation (i.e., no activation function, or `f(z)=z`) or sometimes ReLU if the output must be non-negative.

5.  **Considerations:**
    *   **Vanishing Gradients:** Critical for deep networks. ReLU and its variants are designed to combat this.
    *   **Computational Cost:** ReLU is the cheapest. Sigmoid, Tanh, ELU are more expensive.
    *   **Zero-Centered Output:** Can help with faster convergence (Tanh, ELU in some respects).
    *   **Sparsity:** ReLU and its variants induce sparsity.

**Experimentation:** Often, the best way to choose an activation function is to experiment with different options and see what works best for your specific problem and dataset. Start with common choices and iterate.

This overview should provide a solid understanding of the most common activation functions and their roles in neural networks.

# 02. Neural Networks Basics

This document provides a foundational understanding of Artificial Neural Networks (ANNs), the core components of deep learning systems.

## The Neuron (Perceptron)

The fundamental building block of a neural network is the neuron, also known as a perceptron in simpler architectures. It's a computational unit that receives inputs, processes them, and produces an output.

```
  Input 1 (x1) ---w1---\
  Input 2 (x2) ---w2----[ Neuron ] --- Output (y)
  Input n (xn) ---wn---/    |
                   Bias (b) /
```

Key components of a neuron:

1.  **Inputs (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>):** These are the features or values fed into the neuron from the previous layer or the raw data. Each input has an associated weight.

2.  **Weights (w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>):** Each input connection has a weight associated with it. The weight determines the importance of that input in the neuron's computation. These are the parameters that the network learns during the training process. A positive weight excites the neuron, while a negative weight inhibits it.

3.  **Bias (b):** The bias term is an additional parameter that is added to the weighted sum of inputs. It allows the neuron to be activated even when all inputs are zero, or to shift the activation function. It increases the flexibility of the model to fit the data.

4.  **Weighted Sum (z):** The neuron first calculates a weighted sum of its inputs and adds the bias. This is also called the net input or pre-activation.
    Mathematically:
    `z = (x_1 * w_1) + (x_2 * w_2) + ... + (x_n * w_n) + b`
    Or in vector notation:
    `z = w · x + b`
    where `w` is the vector of weights and `x` is the vector of inputs.

5.  **Activation Function (f or σ or g):** The result of the weighted sum (`z`) is then passed through an activation function. This function introduces non-linearity into the model, allowing the network to learn complex patterns that linear models cannot. The output of the activation function is the final output of the neuron.
    `y = f(z) = f(w · x + b)`

    Common activation functions include:
    *   **Sigmoid:** `σ(z) = 1 / (1 + e^(-z))` - Squashes output between 0 and 1. Used in older networks, especially for binary classification output layers.
    *   **Tanh (Hyperbolic Tangent):** `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))` - Squashes output between -1 and 1. Often preferred over sigmoid in hidden layers.
    *   **ReLU (Rectified Linear Unit):** `ReLU(z) = max(0, z)` - Outputs the input directly if positive, otherwise outputs zero. Very popular due to its simplicity and effectiveness in combating the vanishing gradient problem.
    *   **Leaky ReLU:** `LeakyReLU(z) = max(αz, z)` where `α` is a small constant (e.g., 0.01) - A variant of ReLU that allows a small, non-zero gradient when the unit is not active.
    *   **Softmax:** Often used in the output layer for multi-class classification. Converts a vector of raw scores (logits) into a probability distribution.

## Types of Layers

Neural networks are typically organized into layers of neurons.

1.  **Input Layer:**
    *   The first layer in the network.
    *   It receives the raw input data (features) for the model.
    *   The neurons in the input layer do not perform any computation; they simply pass the data to the first hidden layer.
    *   The number of neurons in the input layer is equal to the number of features in the input data.
    ```
    [Input Feature 1] ---->
    [Input Feature 2] ---->  (To Hidden Layer 1)
    [Input Feature N] ---->
    ```

2.  **Hidden Layers:**
    *   Layers between the input layer and the output layer.
    *   These layers are responsible for learning and extracting features from the input data.
    *   Neurons in hidden layers perform computations (weighted sum + activation).
    *   A network can have zero or more hidden layers. Networks with multiple hidden layers are considered "deep."
    *   The number of hidden layers and the number of neurons in each hidden layer are hyperparameters that are chosen during the model design.
    ```
    (From Input Layer or Previous Hidden Layer)
        |
    [Neuron H1_1] ---\
    [Neuron H1_2] ---- ----> (To Next Hidden Layer or Output Layer)
    [Neuron H1_M] ---/
        |
    (Hidden Layer 1)
    ```

3.  **Output Layer:**
    *   The final layer in the network.
    *   It produces the network's output, which could be a prediction, classification, or some other value.
    *   The number of neurons in the output layer and their activation functions depend on the type of problem:
        *   **Binary Classification:** One neuron with a sigmoid activation function (outputting a probability between 0 and 1).
        *   **Multi-class Classification:** N neurons (where N is the number of classes) with a softmax activation function (outputting a probability distribution across classes).
        *   **Regression:** One neuron with a linear activation function (or no activation function) to output a continuous value.
    ```
    (From Last Hidden Layer)
        |
    [Neuron O1] ----> [Prediction 1]
    [Neuron O2] ----> [Prediction 2 (if applicable)]
    ...
    ```

## Building a Simple Neural Network: Architecture

A neural network architecture defines how neurons are structured in layers and how they are connected.

*   **Feedforward Neural Network:** The simplest type, where information flows in only one direction—from the input layer, through the hidden layers (if any), to the output layer. There are no cycles or loops in the connections.
*   **Example: A simple network for binary classification:**
    *   Input Layer: 2 neurons (e.g., for 2 input features)
    *   Hidden Layer 1: 3 neurons with ReLU activation
    *   Output Layer: 1 neuron with Sigmoid activation

    ```
    Textual Diagram:

    Input 1 ---O
                \
                 --- Neuron H1 (ReLU) ---
                /                         \
    Input 2 ---O                           --- Neuron Out (Sigmoid) --> Prediction
                 --- Neuron H2 (ReLU) --- /
                /                         \
               --- Neuron H3 (ReLU) ---
    (Layer:      Input)    (Hidden)          (Output)
    ```

*   **Connections:** Typically, neurons in one layer are fully connected to neurons in the next layer. This means each neuron in a layer receives input from all neurons in the previous layer, and its output goes to all neurons in the subsequent layer.

## Forward Propagation

Forward propagation (or forward pass) is the process by which input data is fed into the network and travels through its layers to produce an output.

1.  **Input:** The network receives an input vector `X`.
2.  **Layer by Layer Computation:**
    *   For each neuron in the first hidden layer, calculate the weighted sum of its inputs (from the input layer) and apply the activation function. The outputs of this layer become the inputs for the next layer.
    *   Repeat this process for all subsequent hidden layers.
    *   Finally, compute the output of the neurons in the output layer.

**Mathematical Representation (for a single instance):**

Let:
*   `X` be the input vector.
*   `W^(l)` be the weight matrix for layer `l`.
*   `b^(l)` be the bias vector for layer `l`.
*   `a^(l)` be the activation vector (output) of layer `l`.
*   `f^(l)` be the activation function for layer `l`.
*   `z^(l)` be the weighted sum (pre-activation) for layer `l`.

The input layer's activation is just the input features:
`a^(0) = X`

For each subsequent layer `l` (from 1 to L, where L is the output layer):
1.  Calculate the weighted sum:
    `z^(l) = W^(l) * a^(l-1) + b^(l)`
    *   `W^(l)` has dimensions `(number of neurons in layer l, number of neurons in layer l-1)`
    *   `a^(l-1)` has dimensions `(number of neurons in layer l-1, 1)`
    *   `b^(l)` has dimensions `(number of neurons in layer l, 1)`
    *   `z^(l)` has dimensions `(number of neurons in layer l, 1)`

2.  Apply the activation function element-wise:
    `a^(l) = f^(l)(z^(l))`

The final output of the network is `a^(L)`, often denoted as `ŷ` (y-hat).

**Example (1 hidden layer):**
`z^(1) = W^(1) * X + b^(1)`
`a^(1) = f^(1)(z^(1))`
`z^(2) = W^(2) * a^(1) + b^(2)`
`a^(2) = ŷ = f^(2)(z^(2))`

## Backward Propagation and Gradient Descent

After forward propagation, the network's output (`ŷ`) is compared to the true target value (`y`) using a **loss function** (or cost function), `J(W, b)`. The loss function quantifies how "wrong" the network's prediction is. Examples: Mean Squared Error (MSE) for regression, Cross-Entropy Loss for classification.

The goal of training is to find the weights `W` and biases `b` that minimize this loss function.

1.  **Backward Propagation (Backprop):**
    *   This is an algorithm for efficiently computing the gradients of the loss function with respect to each weight and bias in the network.
    *   It works by starting at the output layer and moving backward through the network, applying the chain rule of calculus to calculate how much each parameter contributed to the overall error.
    *   For each layer `l`, it computes `∂J/∂W^(l)` and `∂J/∂b^(l)`.

2.  **Gradient Descent:**
    *   Once the gradients are computed, an optimization algorithm like Gradient Descent is used to update the weights and biases.
    *   The parameters are adjusted in the direction opposite to their gradient, scaled by a **learning rate (α)**.
    *   Update rule for a weight `w_ij^(l)`:
        `w_ij^(l) = w_ij^(l) - α * (∂J / ∂w_ij^(l))`
    *   Update rule for a bias `b_i^(l)`:
        `b_i^(l) = b_i^(l) - α * (∂J / ∂b_i^(l))`
    *   This process (forward pass, loss calculation, backward pass, parameter update) is repeated for many iterations (epochs) over the training dataset.
    *   Variations of Gradient Descent (e.g., Stochastic Gradient Descent (SGD), Mini-batch Gradient Descent, Adam, RMSprop) are commonly used to improve training speed and stability. These will be detailed in a later section.

**Conceptual Overview:**
Imagine the loss function as a hilly landscape. The current set of weights and biases places you at some point on this landscape. Gradient Descent tries to take steps downhill to find the lowest point (minimum loss). The gradient tells you the steepest direction, and the learning rate controls the size of your steps.

## Representational Power of Neural Networks

*   **Universal Approximation Theorem:** A feedforward neural network with a single hidden layer containing a finite number of neurons and a non-linear activation function can approximate any continuous function on compact subsets of R<sup>n</sup> to any desired degree of accuracy. This means that, theoretically, even a simple neural network can represent a wide variety of complex functions, given enough neurons in the hidden layer.
*   **Hierarchical Feature Learning:** Deep neural networks (with multiple hidden layers) are particularly powerful because they can learn hierarchical representations of data.
    *   The first few layers might learn simple, low-level features (e.g., edges, corners in images; basic n-grams in text).
    *   Subsequent layers combine these simpler features to learn more complex, abstract features (e.g., object parts, textures in images; phrases, topics in text).
    *   This ability to automatically learn relevant features is a key advantage of deep learning.

## Different Types of Neural Network Architectures (Brief Mention)

While the feedforward network described above is fundamental, several specialized architectures have been developed for different types of data and tasks:

1.  **Convolutional Neural Networks (CNNs or ConvNets):**
    *   Highly effective for grid-like data, especially images.
    *   Use special layers like convolutional layers (to detect local features) and pooling layers (to downsample and reduce dimensionality).
    *   Key for image classification, object detection, image segmentation.

2.  **Recurrent Neural Networks (RNNs):**
    *   Designed to process sequential data, where the order of information matters (e.g., text, time series, speech).
    *   Have feedback loops, allowing information from previous steps in the sequence to persist and influence current predictions.
    *   Variants like LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units) address challenges in learning long-range dependencies.

These architectures, along with others like Transformers, Autoencoders, and Generative Adversarial Networks (GANs), will be explored in more detail in subsequent sections.

This foundational knowledge of neural network basics is crucial before diving into these more specialized and advanced topics.

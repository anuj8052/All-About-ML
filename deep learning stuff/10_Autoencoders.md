# 10. Autoencoders

Autoencoders are a type of artificial neural network used for unsupervised learning, primarily aimed at learning efficient data codings (representations) of unlabeled data. They achieve this by training the network to reconstruct its own input.

## Introduction to Autoencoders

### Unsupervised Learning Technique

Autoencoders fall under the category of unsupervised learning because they do not require labeled data for training. The network learns by trying to copy its input to its output. While this might sound trivial, the way autoencoders are structured forces them to learn meaningful properties of the data.

### Learning Efficient Data Codings (Representations)

The primary goal of an autoencoder is to learn a compressed representation (encoding) for a set of data. This learned representation, often called the **latent space representation** or **code**, captures the most salient features of the input data.

### Structure: Encoder and Decoder

An autoencoder consists of two main parts:

1.  **Encoder:** This part of the network compresses the input into a lower-dimensional latent space representation.
2.  **Decoder:** This part of the network reconstructs the input data from the latent space representation.

```
Textual Diagram: Basic Autoencoder Architecture

Input (x) ----> [Encoder Network] ----> Latent Space (z) ----> [Decoder Network] ----> Output (x') (Reconstruction of x)
                                         (Bottleneck)
```

### Goal: Reconstruct the Input (Output ≈ Input)

The autoencoder is trained to make the output `x'` as close as possible to the original input `x`. The learning process involves minimizing a **reconstruction loss**, which measures the difference between the input and the output.

### The "Bottleneck" Layer (Latent Space Representation)

A key architectural feature of most autoencoders is a "bottleneck" in the network. This bottleneck is a hidden layer with fewer neurons than the input or output layers. This layer forces the encoder to learn a compressed representation of the input data because it has to squeeze all the necessary information through this narrower passage. This compressed layer is where the **latent space representation (z)** is formed.

If the bottleneck layer had the same or more dimensions than the input, the autoencoder could potentially learn the identity function (simply copying the input to the output) without learning any useful features about the data's structure.

## Encoder

*   **Function:** The encoder maps the input data `x` to a lower-dimensional representation `z` in the latent space.
    `z = e(x)`
    where `e` denotes the encoding function.
*   **Structure:** Typically, the encoder is a feedforward neural network. It consists of one or more hidden layers, with progressively fewer neurons, culminating in the bottleneck layer that produces `z`.
    *   Activation functions (e.g., ReLU, sigmoid, tanh) are used in the hidden layers.
    *   The final layer of the encoder (the bottleneck layer) might use a linear activation or a specific activation depending on the desired properties of the latent space.

```
Textual Diagram: Encoder Structure
Input (x) -> [Hidden Layer 1 (e.g., 128 units)] -> [Hidden Layer 2 (e.g., 64 units)] -> Bottleneck Layer (z) (e.g., 32 units)
```

## Decoder

*   **Function:** The decoder maps the latent space representation `z` back to the original input space, producing a reconstruction `x'`.
    `x' = d(z)`
    where `d` denotes the decoding function.
*   **Structure:** The decoder is also typically a feedforward neural network. Its architecture is often a mirror image of the encoder, with hidden layers that progressively increase in the number of neurons, finally reaching the output layer which has the same number of neurons as the original input.
    *   Activation functions are used in the hidden layers.
    *   The activation function of the output layer depends on the type of input data:
        *   **Sigmoid:** If input values are in the range [0, 1] (e.g., normalized pixel values for binary images or images with pixel intensities scaled to [0,1]).
        *   **Linear:** If input values are continuous and can take any real number (e.g., raw sensor data).

```
Textual Diagram: Decoder Structure
Bottleneck Layer (z) (e.g., 32 units) -> [Hidden Layer 1 (e.g., 64 units)] -> [Hidden Layer 2 (e.g., 128 units)] -> Output (x') (same dimension as input x)
```

## Latent Space (Bottleneck Layer)

*   **Compressed Representation:** The latent space is the compressed version of the input data learned by the encoder. Its dimensionality is usually smaller than the input data's dimensionality.
*   **Dimensionality Reduction Aspect:** By forcing the input data through this lower-dimensional bottleneck, the autoencoder learns to perform dimensionality reduction. It aims to retain the most important information needed to reconstruct the input, effectively discarding noise and redundancy.
*   The properties of the latent space are crucial. A well-trained autoencoder will produce a latent space where similar inputs are mapped to nearby points.

## Training Autoencoders

Autoencoders are trained using backpropagation, similar to other neural networks.

### Loss Function (Reconstruction Error)

The loss function measures how different the reconstructed output `x'` is from the original input `x`. The choice of loss function depends on the nature of the input data:

1.  **Mean Squared Error (MSE):**
    *   Commonly used when the input data is continuous (e.g., pixel values in grayscale or RGB images that are not normalized to [0,1], or other real-valued data).
    *   **Formula:** For an input `x` and reconstruction `x'`:
        `L(x, x') = ||x - x'||^2 = Σ_i (x_i - x'_i)^2`
        (Often averaged over the batch of inputs and the dimensions of the input).
        `MSE = (1/N) * Σ_{j=1}^{N} (1/D) * Σ_{i=1}^{D} (x_{j,i} - x'_{j,i})^2`
        where `N` is the number of samples in a batch, and `D` is the dimensionality of each sample.

2.  **Binary Cross-Entropy (BCE):**
    *   Used when the input data consists of binary values (0 or 1) or values normalized to the range [0, 1] (e.g., binary images, or pixel intensities scaled to [0,1] where the output layer uses a sigmoid activation).
    *   **Formula:** For a single input vector `x` and reconstruction `x'`:
        `L(x, x') = - Σ_i [ x_i * log(x'_i) + (1 - x_i) * log(1 - x'_i) ]`
        (Averaged over the batch).

The goal of training is to find the encoder and decoder parameters (weights and biases) that minimize this reconstruction loss.

## Types of Autoencoders

### 1. Vanilla Autoencoder

This is the basic autoencoder architecture described above, with an encoder, a decoder, and a bottleneck layer, trained to reconstruct the input by minimizing a reconstruction loss.

### 2. Denoising Autoencoders (DAEs)

*   **Concept:** Denoising autoencoders are trained to reconstruct a *clean* version of the input from a *corrupted* version of it.
*   **Training Process:**
    1.  Take an input sample `x`.
    2.  Corrupt it to get `x̃` (x-tilde). Corruption can be adding Gaussian noise, randomly setting some inputs to zero (salt-and-pepper noise), etc.
    3.  Train the autoencoder to reconstruct the original, clean input `x` from the corrupted input `x̃`.
    `z = e(x̃)`
    `x' = d(z)`
    Loss: `L(x, x')` (compare original `x` with reconstruction `x'`).
*   **Benefit:** By forcing the autoencoder to remove noise, it learns more robust features and is less likely to learn the identity function. It has to understand the underlying structure of the data to effectively denoise it.

### 3. Sparse Autoencoders

*   **Concept:** Sparse autoencoders aim to learn representations where only a small number of neurons in the hidden layers (often the bottleneck layer, or other hidden layers) are active at any given time. This means most hidden units output values close to zero.
*   **Regularization:** Sparsity is typically achieved by adding a **sparsity penalty** to the loss function, in addition to the reconstruction loss. This penalty discourages too many neurons from activating.
    *   One common penalty is the KL divergence between the average activation of a neuron over the training batch and a small desired sparsity parameter `ρ` (rho, e.g., 0.05).
    `Loss = Reconstruction_Loss + λ * Σ_j KL(ρ || p̂_j)`
    where `p̂_j` is the average activation of hidden unit `j`, and `λ` controls the weight of the sparsity penalty.
*   **Benefit:** Can lead to learning features that are more specialized and interpretable, as different neurons might become selective for different input patterns. They can have a bottleneck layer with many units but still learn useful features due to the sparsity constraint.

### 4. Variational Autoencoders (VAEs)

*   **Concept:** VAEs are a type of **generative model**, meaning they can generate new data samples that resemble the training data. They are built on probabilistic principles.
*   **Encoder:** Instead of mapping the input to a single point in the latent space, the VAE encoder maps the input `x` to a **probability distribution** over the latent space. This distribution is typically a Gaussian distribution, defined by a mean vector `μ` and a standard deviation (or log-variance) vector `σ`.
    *   Encoder outputs `μ_z = e_μ(x)` and `log(σ_z^2) = e_σ(x)`.
*   **Latent Space Sampling:** A point `z` is then **sampled** from this learned distribution `N(μ_z, σ_z^2I)`.
    `z = μ_z + σ_z * ε` (where `ε` is sampled from a standard normal distribution `N(0, I)` - this is the reparameterization trick).
*   **Decoder:** The decoder takes the sampled latent vector `z` and reconstructs the input `x' = d(z)`, similar to a standard autoencoder.
*   **Loss Function:** VAEs have a more complex loss function consisting of two terms:
    1.  **Reconstruction Loss:** Same as in standard autoencoders (e.g., MSE or BCE), encouraging good reconstruction.
    2.  **Regularization Term (KL Divergence):** This term forces the learned latent distributions to be close to a standard normal distribution `N(0, I)`. This regularizes the latent space, making it smooth and continuous, which is essential for generating new, coherent data.
        `Loss = Reconstruction_Loss + KL(N(μ_z, σ_z^2I) || N(0, I))`
*   **Benefit:** VAEs can generate new data by sampling `z` from the prior distribution `N(0, I)` and passing it through the decoder. The learned latent space is continuous and meaningful.
*   **(High-Level):** While GANs (Generative Adversarial Networks) are another prominent generative model, VAEs offer a probabilistic approach to generation and representation learning.

### 5. Contractive Autoencoders (CAEs)

*   **Concept:** Contractive autoencoders add a penalty term to the loss function that encourages the learned representation (activations in the latent space) to be robust to small changes in the input. That is, it encourages the derivative of the hidden layer activations with respect to the input to be small.
*   **Regularization:** The penalty is the Frobenius norm of the Jacobian matrix of the encoder's activations with respect to the input.
    `Loss = Reconstruction_Loss + λ * ||J_f(x)||_F^2`
*   **Benefit:** This forces the autoencoder to learn features that capture directions of variation in the data that are important for reconstruction, while being insensitive (contracting) to other directions of variation (noise or irrelevant details).

### 6. Stacked Autoencoders (Deep Autoencoders)

*   **Concept:** A stacked autoencoder is simply an autoencoder with multiple hidden layers in both the encoder and decoder (i.e., a "deep" autoencoder).
*   **Structure:**
    *   Encoder: `Input -> Hidden1 -> Hidden2 -> ... -> Bottleneck`
    *   Decoder: `Bottleneck -> HiddenK -> ... -> HiddenN -> Output`
*   **Training:** Can be trained end-to-end like any deep network. Historically, they were sometimes trained layer-wise (greedy layer-wise pre-training): train a shallow autoencoder, then use its learned encodings as input to train another autoencoder, and so on. This approach is less common now with better optimization techniques for deep networks.
*   **Benefit:** Deeper autoencoders can learn more complex hierarchical representations, similar to how deep feedforward networks learn hierarchical features.

## Applications of Autoencoders

Autoencoders have a wide range of applications due to their ability to learn useful data representations:

1.  **Dimensionality Reduction / Feature Learning:**
    *   The encoder part can be used to transform high-dimensional data into a lower-dimensional representation (the latent space). This can be more powerful than traditional methods like PCA because autoencoders can learn non-linear transformations.
    *   The learned features in the latent space can then be used for other tasks, like classification.

2.  **Data Denoising:**
    *   As seen with Denoising Autoencoders, they can be trained to remove noise from corrupted data, effectively learning to separate signal from noise.

3.  **Anomaly Detection / Outlier Detection:**
    *   Autoencoders are trained to reconstruct "normal" data well. When presented with an anomalous or outlier data point (which is significantly different from the training data), the autoencoder will likely have a high reconstruction error because it hasn't learned to represent such data effectively.
    *   By setting a threshold on the reconstruction error, one can identify anomalies.

4.  **Pre-training for Supervised Learning Tasks:**
    *   The encoder part of an autoencoder can be pre-trained on a large amount of unlabeled data to learn good initial feature representations.
    *   These learned encoder weights can then be used to initialize the early layers of a supervised neural network (e.g., a classifier), which is then fine-tuned on a smaller amount of labeled data. This can improve performance, especially when labeled data is scarce.

5.  **Data Generation (especially VAEs):**
    *   Variational Autoencoders can generate new data samples by sampling from the learned latent space and passing these samples through the decoder. This is useful for tasks like generating new images, music, or text.

6.  **Image Compression:** Although not typically state-of-the-art for general compression, the core idea of encoding to a smaller representation and then decoding is akin to compression.

7.  **Recommendation Systems:** Autoencoders can be used to learn latent features of users and items, which can then be used to predict user preferences.

Autoencoders are a versatile tool in the unsupervised learning landscape, offering powerful ways to learn meaningful representations from data without explicit labels. Their various forms allow them to be adapted to a wide range of tasks from simple dimensionality reduction to complex generative modeling.

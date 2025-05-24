# 07. Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a class of neural networks specifically designed to process sequential data. Unlike feedforward networks that process fixed-size inputs, RNNs can handle inputs of varying lengths and leverage information from previous inputs in the sequence to influence current predictions.

## Introduction to RNNs

### Why RNNs? Handling Sequential Data

Many real-world datasets are sequential in nature, meaning the order of elements matters. Examples include:

*   **Time Series Data:** Stock prices, weather measurements over time, sensor readings.
*   **Text Data:** Sequences of characters or words (sentences, documents).
*   **Speech Data:** Sequences of acoustic features.
*   **Video Data:** Sequences of image frames.

Traditional feedforward neural networks (like MLPs and CNNs) assume that inputs are independent of each other. They are not well-suited for tasks where:
1.  The input or output length can vary.
2.  Understanding context from previous elements in the sequence is crucial for making predictions about current or future elements.

RNNs address these limitations by introducing a "memory" mechanism.

### The Concept of "Memory" or Hidden State

The core idea behind RNNs is the use of a **hidden state** (or memory). This hidden state captures information about what has been processed in the sequence so far. At each time step, the RNN updates its hidden state based on the current input and the previous hidden state. This allows information from earlier time steps to persist and influence computations at later time steps.

```
Textual Diagram: Basic RNN Cell
Input (x_t) at time t -----> [RNN Cell] -----> Output (y_t) at time t
                              ^      |
                              |      |
                        Hidden State (h_t)
                              |      |
                              ------- (Loop: h_{t-1} from previous step)
```
The hidden state `h_t` acts as a summary or context of the past sequence elements relevant to the task.

### Applications of RNNs

RNNs and their variants (like LSTMs and GRUs) have been successfully applied to a wide range of tasks involving sequential data:

*   **Language Modeling:** Predicting the next word in a sequence of words. This is fundamental to many NLP tasks.
*   **Machine Translation:** Translating a sentence from one language to another (e.g., English to French).
*   **Speech Recognition:** Converting spoken language into text.
*   **Sentiment Analysis:** Determining the sentiment (e.g., positive, negative, neutral) of a piece of text.
*   **Text Generation:** Generating new text, such as stories, poems, or code.
*   **Image Captioning:** Generating a textual description of an image.
*   **Time Series Prediction:** Forecasting future values in a time series (e.g., stock prices, weather).
*   **Video Analysis:** Classifying activities or understanding content in video frames.
*   **Music Generation:** Composing new musical pieces.

## Structure of an RNN

### Recurrent Loop: How Information Persists

The defining feature of an RNN is its recurrent loop. A neuron or a layer of neurons in an RNN receives input not only from the current external input but also from its own output (or hidden state) from the previous time step.

*   At each time step `t`, the RNN cell takes two inputs:
    1.  The current input data `x_t`.
    2.  The hidden state from the previous time step `h_{t-1}`.
*   It then computes the new hidden state `h_t` and (optionally) an output `y_t`.
*   This `h_t` is then passed to the next time step `t+1`.

### Unrolling an RNN in Time

To understand how RNNs process sequences, it's helpful to visualize them "unrolled" or "unfolded" in time. Unrolling means creating a copy of the RNN cell for each time step in the input sequence. The hidden state is passed from one copy to the next.

```
Textual Diagram: Unrolled RNN
Time:       t=1                  t=2                  t=3
Input:      x_1                  x_2                  x_3
            |                    |                    |
            V                    V                    V
        [RNN Cell] --h_1--> [RNN Cell] --h_2--> [RNN Cell] --h_3-->
            |                    |                    |
            V                    V                    V
Output:     y_1                  y_2                  y_3

(Initial hidden state h_0 is typically a vector of zeros or learned parameters)
```
Each `[RNN Cell]` in the unrolled diagram represents the same set of weights. The weights are shared across all time steps. This parameter sharing is crucial for generalizing to sequences of different lengths and for keeping the model compact.

### Inputs, Hidden States, Outputs at Each Time Step

*   **Input `x_t`:** The data fed into the network at time step `t`. This could be a vector representing a word (e.g., word embedding), a character, or a set of features from a time series.
*   **Hidden State `h_t`:** The internal memory of the network at time step `t`. It's calculated based on the current input `x_t` and the previous hidden state `h_{t-1}`.
    `h_t = f(W_hh * h_{t-1} + W_xh * x_t + b_h)`
    where `f` is typically a non-linear activation function like `tanh` or `ReLU`.
*   **Output `y_t`:** The prediction made by the network at time step `t`. It's usually calculated from the current hidden state `h_t`.
    `y_t = g(W_hy * h_t + b_y)`
    where `g` might be a softmax function (for classification like predicting the next word) or a linear activation (for regression).

### Mathematical Formulation of the Recurrent Update

Let:
*   `x_t` be the input vector at time step `t`.
*   `h_t` be the hidden state vector at time step `t`.
*   `h_{t-1}` be the hidden state vector at the previous time step `t-1`.
*   `y_t` be the output vector at time step `t`.
*   `W_xh` be the weight matrix for connections from input to hidden layer.
*   `W_hh` be the weight matrix for recurrent connections from previous hidden state to current hidden state.
*   `W_hy` be the weight matrix for connections from hidden layer to output layer.
*   `b_h` be the bias vector for the hidden layer.
*   `b_y` be the bias vector for the output layer.

The core computations at each time step `t` are:

1.  **Hidden State Update:**
    `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
    (Tanh is a common choice for the hidden layer activation, though ReLU can also be used.)
    The initial hidden state `h_0` is often initialized to zeros or learned.

2.  **Output Calculation:**
    `y_t = W_hy * h_t + b_y`
    (If the task is classification, `y_t` might be passed through a softmax activation: `ŷ_t = softmax(W_hy * h_t + b_y)`).

The same weight matrices (`W_xh`, `W_hh`, `W_hy`) and biases (`b_h`, `b_y`) are used at every time step.

## Forward Propagation in RNNs

Forward propagation in an RNN involves computing the hidden states and outputs sequentially through the unrolled network.

**Algorithm (Conceptual):**
1.  Initialize the first hidden state `h_0` (e.g., to a vector of zeros).
2.  For each time step `t` from 1 to `T` (where `T` is the sequence length):
    a.  Take the input `x_t` for the current time step.
    b.  Compute the new hidden state:
        `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)`
    c.  Compute the output for the current time step (if needed at each step):
        `y_t = W_hy * h_t + b_y` (or `ŷ_t = softmax(W_hy * h_t + b_y)`)
    d.  Store `h_t` and `y_t` (or `ŷ_t`). The current `h_t` becomes `h_{t-1}` for the next step.
3.  The final output can be `y_T`, all `y_t`'s, or some aggregation of hidden states, depending on the task.

## Backward Propagation Through Time (BPTT)

Training an RNN involves adjusting the shared weights (`W_xh`, `W_hh`, `W_hy`) and biases to minimize a loss function. The loss function typically depends on the outputs `y_t` (or `ŷ_t`) and the true target values.

*   **Conceptual Explanation:**
    BPTT is an extension of the standard backpropagation algorithm used for feedforward networks. Since the RNN is unrolled in time, BPTT involves:
    1.  **Forward Pass:** Process the input sequence as described above, computing all hidden states and outputs, and calculate the loss (e.g., sum of losses at each time step where a prediction is made).
    2.  **Backward Pass:** The error is propagated backward through the unrolled network, from the last time step to the first.
    3.  **Gradient Accumulation:** Because the weights are shared across all time steps, the gradients for a given weight matrix (e.g., `W_hh`) are calculated at each time step and then summed up (or averaged) across all time steps to get the total gradient for that weight.
        `∂L/∂W = Σ_t (∂L_t / ∂W)` where `L` is the total loss and `L_t` is the loss at time step `t`.
    4.  **Parameter Update:** The weights are then updated using an optimization algorithm (e.g., SGD, Adam) based on these accumulated gradients.

*   **Challenges: Vanishing and Exploding Gradients in RNNs:**
    During BPTT, gradients are propagated backward through many layers (equal to the number of time steps). This can lead to two significant problems:
    1.  **Vanishing Gradients:**
        *   If the recurrent weight matrix `W_hh` has eigenvalues less than 1 (or activation function derivatives are small), gradients can shrink exponentially as they are propagated back through time.
        *   This means that the influence of earlier inputs (`x_1, x_2, ...`) on the current output `y_t` becomes negligible for long sequences.
        *   The network struggles to learn long-range dependencies (dependencies between elements far apart in the sequence).
    2.  **Exploding Gradients:**
        *   If the recurrent weight matrix `W_hh` has eigenvalues greater than 1, gradients can grow exponentially as they are propagated back.
        *   This can lead to very large weight updates and unstable training (often resulting in NaN values).
        *   Exploding gradients are easier to detect and can often be mitigated by techniques like **gradient clipping** (scaling down gradients if their norm exceeds a threshold).

Vanishing gradients are a more persistent problem and were a major motivation for developing more sophisticated RNN architectures like LSTMs and GRUs.

## Types of RNNs (based on input/output sequences)

RNNs can be categorized based on the nature of their input and output sequences:

1.  **One-to-one (Standard NN, for context):**
    *   Input: Single fixed-size input.
    *   Output: Single fixed-size output.
    *   This is the structure of a traditional feedforward neural network (no sequence involved).
    *   Example: Image classification (one image in, one class label out).

2.  **One-to-many (Sequence Output):**
    *   Input: Single fixed-size input.
    *   Output: Sequence of outputs.
    *   The input might be fed at the first step, or fed to each step while generating output.
    *   Example: Image captioning (one image in, a sequence of words out).
    ```
    Textual Diagram: One-to-Many
    Input (x) --> [RNN Cell] --h_1--> [RNN Cell] --h_2--> [RNN Cell] --h_3--> ...
                    |                    |                    |
                    V                    V                    V
                  Output(y_1)          Output(y_2)          Output(y_3)
    ```

3.  **Many-to-one (Sequence Input):**
    *   Input: Sequence of inputs.
    *   Output: Single fixed-size output (usually after processing the entire sequence).
    *   Example: Sentiment analysis (a sentence/review in, a single sentiment score out), text classification.
    ```
    Textual Diagram: Many-to-One
    Input(x_1) --> [RNN Cell] --h_1--> Input(x_2) --> [RNN Cell] --h_2--> Input(x_3) --> [RNN Cell] --h_3--> Output(y)
                                                                                                (e.g. from h_3)
    ```

4.  **Many-to-many (Sequence Input and Output):**
    *   **Delayed (Encoder-Decoder):**
        *   Input: Sequence of inputs.
        *   Output: Sequence of outputs, typically generated after the entire input sequence has been processed.
        *   Example: Machine translation (source sentence in, target sentence out).
        *   Often uses an **Encoder-Decoder architecture**:
            *   **Encoder RNN:** Processes the input sequence and compresses it into a fixed-size context vector (often the final hidden state).
            *   **Decoder RNN:** Takes the context vector and generates the output sequence, one element at a time.
        ```
        Textual Diagram: Many-to-Many (Delayed - Encoder-Decoder)
        Input(x_1) -> [Enc] -> Input(x_2) -> [Enc] -> Input(x_3) -> [Enc] -> Context Vector (h_enc_3)
                                                                              |
                                                                              V
                                                                    [Dec] -> Output(y_1) -> [Dec] -> Output(y_2) -> ...
        ```
    *   **Real-time (Synchronized):**
        *   Input: Sequence of inputs.
        *   Output: Sequence of outputs, with an output generated at each time step corresponding to the input at that step.
        *   Example: Video frame classification (classify each frame of a video).
        ```
        Textual Diagram: Many-to-Many (Real-time)
        Input(x_1) --> [RNN Cell] --h_1--> Input(x_2) --> [RNN Cell] --h_2--> Input(x_3) --> [RNN Cell] --h_3--> ...
                        |                                 |                                 |
                        V                                 V                                 V
                      Output(y_1)                       Output(y_2)                       Output(y_3)
        ```

## Simple RNN (Elman Network) and Jordan Network

These are early and basic RNN architectures:

*   **Elman Network (Simple RNN):**
    *   The hidden layer activations from the previous time step are fed back as input to the hidden layer at the current time step. This is the most common type of "simple RNN" discussed today.
    *   `h_t = f(W_xh * x_t + W_hh * h_{t-1} + b_h)`

*   **Jordan Network:**
    *   The output layer activations from the previous time step are fed back as input to the hidden layer at the current time step.
    *   `h_t = f(W_xh * x_t + W_yh * y_{t-1} + b_h)`
    *   Less common than Elman networks.

## Limitations of Simple RNNs

While conceptually powerful, simple RNNs (Elman networks) have significant limitations:

1.  **Difficulty in Learning Long-Range Dependencies:**
    *   This is the most critical issue. Due to the vanishing gradient problem, the influence of inputs from early time steps on the outputs at much later time steps diminishes rapidly.
    *   The network effectively "forgets" information from the distant past, making it hard to model dependencies that span long intervals in the sequence (e.g., understanding the relationship between the beginning and end of a long paragraph).

2.  **Exploding Gradients:**
    *   Though often manageable with gradient clipping, this can still make training unstable.

These limitations spurred the development of more sophisticated RNN architectures.

## Solutions: LSTMs and GRUs (Brief Mention)

To address the vanishing gradient problem and better capture long-range dependencies, more advanced RNN cell structures were introduced:

*   **Long Short-Term Memory (LSTM):** Incorporates "gates" (input, forget, output gates) and a separate cell state to control the flow of information, allowing the network to selectively remember or forget information over long periods.
*   **Gated Recurrent Unit (GRU):** A simpler alternative to LSTM, also using gates (update and reset gates) to manage information flow, but with fewer parameters.

These architectures (LSTMs and GRUs) have become the standard for most tasks where RNNs are applied, as they are much more effective at learning long-term dependencies. They will be detailed in a subsequent section.

This overview provides a foundational understanding of Recurrent Neural Networks, their structure, operation, and limitations, setting the stage for more advanced recurrent models.

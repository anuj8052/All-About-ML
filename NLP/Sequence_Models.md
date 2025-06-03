# Sequence Models in NLP

## What are Sequence Models?

### Definition:
Sequence models are a class of machine learning models designed to operate on **sequential data**, where the order of elements is crucial. In such data, elements are not independent and identically distributed (i.id.); rather, the occurrence of an element depends on the elements that preceded or follow it.

Examples of sequential data include:
-   **Text:** Sequences of characters, words, or sentences.
-   **Speech:** Sequences of audio frames.
-   **Time Series Data:** Stock prices over time, weather measurements.
-   **DNA sequences:** Sequences of nucleotides.

### Why they are important for NLP:
Natural language is inherently sequential. The meaning of a word often depends on its surrounding words (context), and the structure of sentences follows grammatical rules that dictate word order. Sequence models are therefore fundamental to NLP for tasks such as:
-   Understanding the contextual meaning of words.
-   Generating coherent and grammatically correct text.
-   Translating text from one language to another, preserving meaning and fluency.
-   Classifying text based on its overall sentiment or topic.

## Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a type of neural network specifically designed to process sequences. They maintain an internal "memory" or **hidden state** that captures information about the preceding elements in the sequence.

### Basic Architecture:
-   **Hidden State (h<sub>t</sub>):** At each time step `t`, the RNN computes a hidden state `h_t`. This state is a function of the current input `x_t` at that time step and the hidden state `h_{t-1}` from the previous time step. This recurrence allows information to persist.
-   **Recurrence Relation:** The core of an RNN is its recurrence relation:
    `h_t = f(W_hh * h_{t-1} + W_xh * x_t + b_h)`
    Where:
    -   `h_t` is the hidden state at time step `t`.
    -   `h_{t-1}` is the hidden state at the previous time step `t-1`.
    -   `x_t` is the input at the current time step `t`.
    -   `W_hh` is the weight matrix for recurrent connections (hidden-to-hidden).
    -   `W_xh` is the weight matrix for input connections (input-to-hidden).
    -   `b_h` is the bias vector for the hidden layer.
    -   `f` is an activation function (e.g., tanh or ReLU).
-   An output `y_t` can also be computed at each time step, often from the hidden state:
    `y_t = g(W_hy * h_t + b_y)`
    Where `W_hy` is the hidden-to-output weight matrix, `b_y` is the output bias, and `g` is an activation function (e.g., softmax for classification).

-   **Diagram of an unrolled RNN:**
    An RNN can be visualized as a series of interconnected neural network units, one for each time step in the sequence. The same set of weights (`W_hh`, `W_xh`, `W_hy`) is shared across all time steps.
    ```
    Input:    x_0       x_1       x_2       ...       x_t
              |         |         |                   |
              V         V         V                   V
    RNN Cell [h_0] --> [h_1] --> [h_2] --> ... --> [h_t]  (Hidden states are passed along)
              |         |         |                   |
              V         V         V                   V
    Output:   y_0       y_1       y_2       ...       y_t
    ```
    *(Imagine arrows connecting `h_{t-1}` to the computation of `h_t` within each cell block)*

### Input and Output Types:
RNNs are versatile and can handle various types of sequence mapping tasks:
-   **One-to-one:** Standard neural network, not typically an RNN (e.g., image classification). `Input -> Output`
-   **One-to-many (Sequence Output):** Takes a single input and generates a sequence of outputs (e.g., image captioning: image -> sequence of words; music generation: genre -> sequence of notes). `Input -> Output_1, Output_2, ...`
-   **Many-to-one (Sequence Input):** Takes a sequence of inputs and produces a single output (e.g., sentiment classification: sequence of words -> sentiment; text classification). `Input_1, Input_2, ... -> Output`
-   **Many-to-many:**
    -   **Synchronous (Aligned Input-Output):** Takes a sequence of inputs and generates a corresponding sequence of outputs, where each output aligns with an input (e.g., Part-of-Speech tagging: word -> POS tag; video frame classification). `Input_1 -> Output_1, Input_2 -> Output_2, ...`
    -   **Asynchronous (Delayed Input-Output / Encoder-Decoder):** Takes a sequence of inputs and generates a sequence of outputs, but the lengths of input and output sequences can differ, and there isn't always a direct alignment (e.g., machine translation: English sentence -> French sentence; question answering). This often uses an Encoder-Decoder architecture (see below). `Input_1, Input_2, ... -> Output_1, Output_2, ...`

### Training RNNs:
-   **Backpropagation Through Time (BPTT):** RNNs are trained using a variation of backpropagation called BPTT. The network is "unrolled" for the length of the input sequence, and then standard backpropagation is applied to calculate gradients and update weights. Because weights are shared across time steps, the gradients for a given weight are summed up across all time steps it affects.

### Challenges with Simple RNNs (Vanilla RNNs):
-   **Vanishing Gradient Problem:** During BPTT, gradients can become exponentially smaller as they propagate backward through many time steps. This means that the influence of earlier inputs on the learning of weights for later time steps becomes negligible. Consequently, simple RNNs struggle to learn **long-range dependencies** (dependencies between elements far apart in a sequence).
-   **Exploding Gradient Problem:** Conversely, gradients can also become exponentially larger, leading to unstable training (oscillating or diverging weights). This can often be mitigated by gradient clipping (scaling down gradients if they exceed a threshold).

### Applications of RNNs in NLP:
Despite their limitations, simple RNNs laid the groundwork for more advanced models and have been used in:
-   Language Modeling
-   Text Generation
-   Machine Translation (as part of early Encoder-Decoder models)
-   Sentiment Analysis
-   Named Entity Recognition

## Long Short-Term Memory (LSTM)

LSTMs were specifically designed by Hochreiter & Schmidhuber (1997) to address the vanishing gradient problem and better capture long-range dependencies.

### Motivation:
To allow gradients to flow unchanged over many time steps, LSTMs introduce a more complex internal structure with a dedicated **cell state** and **gating mechanisms**.

### Architecture:
An LSTM cell maintains two main states:
-   **Cell State (C<sub>t</sub>):** This acts as the "long-term memory" of the network. Information can be added to or removed from the cell state, regulated by gates. It flows through the LSTM chain with only minor linear transformations, helping to preserve information over long sequences.
-   **Hidden State (h<sub>t</sub>):** This acts as the "short-term memory" and is also the output for the current time step. Its computation is influenced by the cell state and gates.

LSTMs use three primary **gates** (sigmoid neural network layers that output values between 0 and 1) to control the flow of information:
1.  **Forget Gate (f<sub>t</sub>):** Decides what information to discard from the cell state `C_{t-1}`. It looks at `h_{t-1}` (previous hidden state) and `x_t` (current input).
    `f_t = σ(W_f * [h_{t-1}, x_t] + b_f)`
2.  **Input Gate (i<sub>t</sub>):** Decides which new information to store in the cell state. It has two parts:
    -   A sigmoid layer (input gate) decides which values to update.
        `i_t = σ(W_i * [h_{t-1}, x_t] + b_i)`
    -   A tanh layer creates a vector of new candidate values `Ĉ_t` that could be added to the state.
        `Ĉ_t = tanh(W_C * [h_{t-1}, x_t] + b_C)`
    The cell state is then updated:
    `C_t = f_t * C_{t-1} + i_t * Ĉ_t`
3.  **Output Gate (o<sub>t</sub>):** Decides what to output from the cell state.
    -   A sigmoid layer decides which parts of the cell state to output.
        `o_t = σ(W_o * [h_{t-1}, x_t] + b_o)`
    -   The cell state is passed through tanh (to push values between -1 and 1) and then multiplied by the output of the sigmoid gate.
        `h_t = o_t * tanh(C_t)`

-   **Diagrams of LSTM cell and gate operations:**
    *(Visualizations of an LSTM cell show these gates and the cell state connected. Chris Olah's blog "Understanding LSTM Networks" is a great resource.)*

### Advantages over simple RNNs:
-   Effectively mitigate the vanishing gradient problem.
-   Can learn and remember information over long sequences.

## Gated Recurrent Unit (GRU)

GRUs (Cho et al., 2014) are a variation of LSTMs that simplify the architecture.

### Motivation:
Achieve similar performance to LSTMs with fewer parameters and potentially faster training.

### Architecture:
GRUs have two gates and no separate cell state (the hidden state `h_t` serves both purposes):
1.  **Update Gate (z<sub>t</sub>):** Combines the roles of LSTM's forget and input gates. It decides how much of the previous hidden state to keep and how much of the new candidate hidden state to incorporate.
    `z_t = σ(W_z * [h_{t-1}, x_t] + b_z)`
2.  **Reset Gate (r<sub>t</sub>):** Determines how much of the previous hidden state to ignore when computing the candidate hidden state.
    `r_t = σ(W_r * [h_{t-1}, x_t] + b_r)`
The candidate hidden state `ĥ_t` is:
`ĥ_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)`
The final hidden state `h_t` is:
`h_t = (1 - z_t) * h_{t-1} + z_t * ĥ_t`

### Comparison with LSTM:
-   Fewer parameters.
-   Often similar performance, but can vary by task. GRUs might train faster.

## Bidirectional RNNs (Bi-RNNs, Bi-LSTMs, Bi-GRUs)

### Concept:
Process the sequence in both forward (left-to-right) and backward (right-to-left) directions. The outputs from both directions are then combined.

### How it allows capturing context from both past and future words:
This allows the hidden state at any point `t` to contain information from both preceding and succeeding elements in the sequence.
`h_t = [→h_t ; ←h_t]` (concatenation of forward and backward hidden states)

### Applications: NER, sentiment analysis, POS tagging.

## Stacked/Deep RNNs

### Concept:
Multiple layers of RNNs, where the output of one layer becomes the input to the next.
This allows learning hierarchical representations of the sequence data.

## Encoder-Decoder Architecture (Seq2Seq)

### Concept:
Designed for sequence-to-sequence tasks where input and output sequences can have different lengths (e.g., machine translation, text summarization).
1.  **Encoder:** An RNN (LSTM/GRU) that processes the input sequence and compresses it into a fixed-size **context vector** (often the final hidden state of the encoder). This vector aims to summarize the input sequence.
2.  **Decoder:** Another RNN (LSTM/GRU) that takes the context vector as input (often as its initial hidden state) and generates the output sequence step by step. The output from the previous time step is fed as input to the current time step in the decoder.

### Applications:
-   **Machine Translation:** English sentence (input) -> French sentence (output).
-   **Text Summarization:** Long article (input) -> Short summary (output).
-   **Question Answering:** Question (input) -> Answer (output).
-   **Chatbots:** User utterance (input) -> Bot response (output).

A limitation of the basic Seq2Seq model is that the fixed-size context vector can be a bottleneck for long input sequences. The **Attention Mechanism** (covered in `Attention_Mechanisms.md`) was introduced to address this.

## Practical Considerations

-   **Choosing between LSTM and GRU:** There's no definitive answer. LSTMs are a good default, but GRUs can be faster and perform comparably on some tasks. Experimentation is key.
-   **Importance of Pre-trained Embeddings:** Using pre-trained word embeddings (like Word2Vec, GloVe, FastText) as the initial input layer to the RNN significantly boosts performance, especially with limited task-specific data.
-   **Regularization:** Techniques like Dropout are crucial to prevent overfitting in RNNs, especially LSTMs and GRUs which have many parameters. Apply dropout between stacked RNN layers, but be cautious applying it directly to recurrent connections within an LSTM/GRU cell (variants like `recurrent_dropout` exist).
-   **Gradient Clipping:** Often used to prevent the exploding gradient problem during training.
-   **Padding:** Input sequences in a batch often have different lengths. They need to be padded to a uniform length. Use masking layers to ensure the model ignores these padded values.
-   **Batching:** Grouping sequences by length (or using bucketing) can minimize the amount of padding and improve training efficiency.

## Libraries for Sequence Models

Modern deep learning frameworks provide extensive support for building and training sequence models:

-   **TensorFlow (with Keras API):**
    -   `tf.keras.layers.RNN`
    -   `tf.keras.layers.LSTM`
    -   `tf.keras.layers.GRU`
    -   `tf.keras.layers.Bidirectional` wrapper.
    -   Easy to stack layers and build complex models.
-   **PyTorch:**
    -   `torch.nn.RNN`
    -   `torch.nn.LSTM`
    -   `torch.nn.GRU`
    -   Modules for handling sequences, packing/padding, and efficient batching.

These libraries abstract away many of the low-level details, allowing researchers and practitioners to focus on model architecture and experimentation.
```

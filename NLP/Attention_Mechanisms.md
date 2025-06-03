# Attention Mechanisms in NLP

## What is Attention?

### Intuition:
Humans, when processing information (like reading a text or listening to speech), don't give equal weight to every piece of information. Instead, we **focus on specific parts** that are relevant to the task at hand, while largely ignoring others. For example, when answering a question about a story, you'd pay more attention to the sentences containing information pertinent to that question.

### In NLP:
In the context of Natural Language Processing, particularly in deep learning models, **attention is a mechanism that allows a model to dynamically focus on relevant parts of the input sequence when producing an output at a particular time step.** It mimics this human cognitive ability, enabling the model to selectively weigh the importance of different input elements.

### Addressing the Bottleneck of Fixed-Size Context Vectors:
Before attention, sequence-to-sequence (Seq2Seq) models, like those used for machine translation, typically relied on an **encoder** to compress the entire input sequence into a single, fixed-size **context vector**. This context vector was then passed to the **decoder** to generate the output sequence.

This fixed-size context vector became a **bottleneck** for several reasons:
-   It's difficult to cram all the information from a long input sequence into a single vector, leading to information loss.
-   The decoder had to generate the entire output based only on this single summary, regardless of which part of the output it was currently generating.
-   Performance degraded significantly for long input sequences.

Attention mechanisms were introduced to overcome this limitation by allowing the decoder to "look back" at different parts of the input sequence (encoder hidden states) at each step of generating the output, and decide which parts are most relevant.

## Attention in Encoder-Decoder (Seq2Seq) Models

Attention was first prominently introduced in the context of Neural Machine Translation by Bahdanau et al. (2014) and Luong et al. (2015).

### Core Idea:
-   Instead of the encoder producing a single context vector, it produces a sequence of hidden states (e.g., `h'_1, h'_2, ..., h'_N` for an input sequence of length N).
-   At each decoding step `t` (when generating the `t`-th output word), the decoder uses an attention mechanism to compute a **context vector `c_t`**.
-   This `c_t` is a **weighted sum of all the encoder hidden states `h'_s`**. The weights (attention weights) determine how much focus to put on each encoder hidden state `h'_s` when generating the current output word.
-   This context vector `c_t` is then used, along with the decoder's own hidden state and the previously generated word, to predict the next word in the output sequence.

### General Attention Mechanism Steps:

Let `h_t` be the current hidden state of the decoder (or a query vector) and `h'_s` be the hidden states from the encoder (or key-value pairs).

1.  **Calculate Alignment Scores (Energy Scores), `e_ts`**:
    -   For the decoder's current state (or previous state, depending on the attention type) `h_t` and each encoder hidden state `h'_s`, compute an alignment score `e_ts`. This score quantifies how well the input around position `s` and the output at position `t` match or how relevant `h'_s` is to `h_t`.
    -   **Common scoring functions:**
        -   **Dot Product:** `score(h_t, h'_s) = h_t^T h'_s`
            - Assumes `h_t` and `h'_s` have the same dimensionality.
        -   **Scaled Dot Product (used in Transformers):** `score(h_t, h'_s) = (h_t^T h'_s) / sqrt(d_k)`
            - Where `d_k` is the dimension of the keys (and queries). Scaling prevents overly large dot products for high dimensions, which could push softmax into regions with small gradients.
        -   **General (Bilinear - Luong's "general" style):** `score(h_t, h'_s) = h_t^T W_a h'_s`
            - `W_a` is a trainable weight matrix.
        -   **Concatenation (Additive/Bahdanau-style):** `score(h_t, h'_s) = v_a^T tanh(W_a[h_t ; h'_s])`
            - `[h_t ; h'_s]` denotes concatenation of the two vectors.
            - `W_a` and `v_a` are trainable weight matrices/vectors. This function is a small feed-forward neural network.

2.  **Compute Attention Weights (Alignment Vector), `α_ts`**:
    -   The alignment scores `e_ts` are passed through a **softmax** function to obtain attention weights `α_ts`. These weights form a probability distribution, meaning they are all non-negative and sum to 1 over all source positions `s`.
    -   `α_ts = softmax(e_ts) = exp(e_ts) / (Σ_{k=1}^{N} exp(e_tk))`
    -   `α_ts` represents the importance of encoder hidden state `h'_s` for generating the output at decoder step `t`.

3.  **Compute Context Vector, `c_t`**:
    -   The context vector `c_t` for decoder time step `t` is computed as a **weighted sum** of all encoder hidden states, using the attention weights `α_ts`.
    -   `c_t = Σ_{s=1}^{N} α_ts * h'_s`
    -   This `c_t` captures the relevant information from the input sequence needed at the current decoding step.

4.  **Use Context Vector in Decoder**:
    -   The computed context vector `c_t` is then used by the decoder to predict the next output token `y_t`.
    -   Typically, `c_t` is concatenated with the decoder's current hidden state `h_t` (if `h_t` wasn't used to compute scores, or a modified version of it) and sometimes the embedding of the previously predicted word. This combined vector is then fed through a linear layer and a softmax function to predict the next word.
    -   `P(y_t | y_{<t}, X) = softmax(W_o [h_t ; c_t])` (simplified example)

### Bahdanau Attention (Additive Attention)
-   Proposed by Bahdanau, Cho, and Bengio (2014) for machine translation.
-   **Scoring Function:** Uses the concatenation-based (additive) approach:
    `score(s_{t-1}, h'_s) = v_a^T tanh(W_1 s_{t-1} + W_2 h'_s)`
    -   Note: It typically uses the decoder's *previous* hidden state `s_{t-1}` (from the RNN cell before the current prediction) and the encoder hidden states `h'_s`. `W_1` and `W_2` are weight matrices.
-   The context vector `c_t` is then concatenated with the decoder's previous hidden state `s_{t-1}` and the previous output `y_{t-1}` to compute the current decoder state `s_t`, which then predicts `y_t`.
-   Often associated with using a Bidirectional RNN for the encoder.

### Luong Attention (Multiplicative/Dot-Product Attention)
-   Proposed by Luong, Pham, and Manning (2015).
-   **Scoring Functions:** Introduced several simpler scoring functions:
    -   **Dot:** `score(h_t, h'_s) = h_t^T h'_s`
    -   **General:** `score(h_t, h'_s) = h_t^T W_a h'_s`
    -   **Concat (their version):** `score(h_t, h'_s) = W_a [h_t ; h'_s]` (followed by `v_a^T tanh(...)` if not directly used) - this is less common than their dot/general.
-   It typically uses the decoder's *current* hidden state `h_t` (the output of the RNN cell at the current time step) to compute alignment scores with encoder hidden states `h'_s`.
-   **Combining Context Vector:** After computing `c_t`, Luong et al. proposed concatenating `c_t` with `h_t` and then passing it through a linear layer and a tanh activation to get an "attentional hidden state" `ĥ_t = tanh(W_c [c_t ; h_t])`. This `ĥ_t` is then used for prediction.

## Types of Attention

### 1. Self-Attention (Intra-Attention)
-   **Concept:** An attention mechanism where the model attends to different positions within the *same* sequence (either the input sequence or the output sequence, independently). It calculates the relevance of each word in a sequence with respect to all other words in that same sequence.
-   **How it works:** Each word in the sequence acts as a query, key, and value (or projections of them). The attention scores are computed between each pair of words in the sequence.
-   **Purpose:** Allows the model to capture internal dependencies or structure within a single sequence, such as long-range dependencies, anaphora resolution, or syntactic relationships.
-   **Example:** In the sentence "The animal didn't cross the street because *it* was too tired," self-attention can help the model learn that "it" refers to "the animal."
-   **Key Component of Transformers:** Self-attention is the core mechanism behind the Transformer model, which has revolutionized many NLP tasks.

### 2. Global vs. Local Attention (Luong et al.)
-   **Global Attention:**
    -   This is what has been described for Bahdanau and most Luong examples: the decoder attends to **all** hidden states of the encoder.
    -   **Pros:** Considers all parts of the input.
    -   **Cons:** Can be computationally expensive for very long input sequences.
-   **Local Attention:**
    -   The decoder attends to only a **subset (a window) of encoder hidden states** at each decoding step.
    -   The model first predicts an aligned position `p_t` in the source sequence for the current target word.
    -   A window of size `D` is centered around `p_t`, and attention is computed only over encoder states within this window.
    -   **Pros:** Reduces computational cost, can be more efficient for long sequences.
    -   **Cons:** Might miss relevant information if it falls outside the chosen window. The choice of window size `D` and how to predict `p_t` are important design decisions.

### 3. Hard vs. Soft Attention
-   **Soft Attention:**
    -   This is the most common type and what we've discussed so far.
    -   The attention mechanism computes a weighted average over all source hidden states (or a window).
    -   The weights are "soft" probabilities, and the entire process is **differentiable**, meaning it can be trained end-to-end using standard backpropagation.
-   **Hard Attention:**
    -   Instead of a weighted average, hard attention **selects one specific part** (e.g., one encoder hidden state) of the input sequence to attend to at each step.
    -   This selection is often treated as a categorical choice.
    -   **Pros:** Can be more computationally efficient at inference time.
    -   **Cons:** The process is typically non-differentiable, requiring more complex training techniques like reinforcement learning (e.g., policy gradients) or variational methods. Less common in mainstream NLP tasks compared to soft attention.

### 4. Multi-Head Attention
-   **Brief Mention:** (Details will be covered in `Transformers.md`)
-   **Concept:** Instead of performing a single attention calculation, multi-head attention runs multiple attention mechanisms ("heads") in parallel.
-   **Process:**
    1.  The input queries, keys, and values are linearly projected into different lower-dimensional subspaces `h` times (for `h` heads).
    2.  Scaled dot-product attention is performed independently for each projected version.
    3.  The outputs of these `h` heads are concatenated and then linearly projected again to produce the final output.
-   **Benefit:** Allows the model to **jointly attend to information from different representation subspaces at different positions**. For example, one head might focus on syntactic relationships, while another focuses on semantic similarity.

## Benefits of Attention Mechanisms

-   **Improved Performance:** Significantly boosted performance in tasks like machine translation, text summarization, question answering, and more.
-   **Better Handling of Long Sequences:** By allowing the model to selectively focus on relevant parts of the input, attention mitigates the information bottleneck of fixed-size context vectors, leading to much better performance on long sequences.
-   **Provides Interpretability:** The attention weights (`α_ts`) can be visualized to understand which parts of the input sequence the model was focusing on when generating a particular part of the output. This offers insights into the model's decision-making process.
-   **Flexibility:** Can be applied in various ways (self-attention, encoder-decoder attention) and with different scoring functions.

## Visualizing Attention

Attention weights can be visualized as a **heatmap**, where one axis represents the input sequence and the other represents the output sequence. The intensity of the color at the intersection of an input word and an output word indicates the attention weight – how much the model focused on that input word when generating that output word. This is particularly insightful for machine translation.

*(Example: A diagram showing an English sentence on the x-axis and its French translation on the y-axis, with shaded cells indicating high attention weights between aligned words.)*

## Applications Beyond Seq2Seq

While attention gained prominence in Seq2Seq models, its principles have been successfully applied to a wider range of NLP and even non-NLP tasks:

-   **Text Classification:** Self-attention can help weigh the importance of different words in a sentence for classification (e.g., for sentiment analysis).
-   **Recommendation Systems:** Attending to different items a user has interacted with to predict future preferences.
-   **Computer Vision:** Attention mechanisms can be used in image captioning to focus on different regions of an image when generating words, or in image classification to highlight salient object parts.
-   **Graph Neural Networks:** Attention can be used to weigh the importance of different neighbors when aggregating information in a graph.

Attention has become a foundational concept in modern deep learning, particularly with the rise of Transformer models that are built almost entirely on attention mechanisms.
```

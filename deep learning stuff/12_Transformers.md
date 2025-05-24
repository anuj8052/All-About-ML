# 12. Transformer Networks

Transformer networks have revolutionized the field of deep learning, particularly in Natural Language Processing (NLP), and are increasingly being applied to other domains like computer vision. They were introduced in the seminal paper "Attention Is All You Need" by Vaswani et al. (2017).

## Introduction to Transformers

### "Attention Is All You Need" Paper

The Transformer model was proposed as an alternative to recurrent neural network (RNN) based architectures like LSTMs and GRUs for sequence-to-sequence tasks, such as machine translation. The paper's title itself highlights the central mechanism of the Transformer: **attention**.

### Overcoming Limitations of RNNs/LSTMs

Traditional RNNs and LSTMs process sequences token by token in a sequential manner. This leads to several limitations:

1.  **Sequential Computation Hinders Parallelization:** The recurrent nature means that computation for a time step `t` depends on the hidden state from time step `t-1`. This makes it difficult to parallelize computations within a sequence, slowing down training on long sequences.
2.  **Difficulty with Long-Range Dependencies:** While LSTMs and GRUs were designed to mitigate the vanishing gradient problem, capturing very long-range dependencies (relationships between tokens far apart in a sequence) can still be challenging. The information has to be passed through many intermediate steps, potentially leading to loss or noise.

Transformers address these issues by dispensing with recurrence altogether and relying entirely on an attention mechanism to draw global dependencies between input and output.

### Parallel Processing Capabilities

Since Transformers do not rely on sequential processing of tokens (once input embeddings are formed), they can process all tokens in a sequence simultaneously. The attention mechanism can directly compute relationships between any two tokens in the sequence, regardless of their distance. This allows for significantly more parallelization during training and inference, making it feasible to train on much larger datasets and build larger models.

### Dominance in NLP and Beyond

Transformers quickly became the dominant architecture for NLP tasks, leading to state-of-the-art results in:
*   Machine Translation
*   Text Summarization
*   Question Answering
*   Sentiment Analysis
*   Language Modeling (e.g., BERT, GPT)

Their success has also inspired applications in other fields:
*   **Computer Vision:** Vision Transformer (ViT) models apply Transformers directly to image patches for image classification.
*   **Reinforcement Learning:** Used for modeling trajectories or policies.
*   **Biology:** Protein structure prediction (e.g., AlphaFold 2).

## Core Architecture of Transformers

The original Transformer model proposed an **encoder-decoder architecture**, primarily for sequence-to-sequence tasks like machine translation.

```
Textual Diagram: Basic Transformer Encoder-Decoder Architecture

Input Sequence ----> [Input Embedding + Positional Encoding] ----> [Encoder Stack (N layers)] ----> Encoder Output (K, V)
                                                                             ^                             |
                                                                             |                             |
Target Sequence (shifted right) -> [Output Embedding + Positional Encoding] -> [Decoder Stack (N layers)] --+--> Output Probabilities -> Output Sequence
```

However, variations of this architecture have emerged:

*   **Encoder-Decoder:** Used for tasks where an input sequence is transformed into an output sequence (e.g., machine translation, summarization).
*   **Encoder-only (e.g., BERT):** Used for tasks that require understanding the input sequence, such as text classification, sentiment analysis, or named entity recognition. The output of the encoder provides rich contextual representations of the input tokens.
*   **Decoder-only (e.g., GPT):** Used for generative tasks, such as language modeling (predicting the next word in a sequence) or text generation. These models are auto-regressive.

## Key Components

### 1. Self-Attention Mechanism

Self-attention is the heart of the Transformer. It allows the model to weigh the importance of different tokens in an input sequence when processing each token. For a given token, self-attention computes how much focus to place on all other tokens in the same sequence (including itself) to better encode or predict it.

#### Queries (Q), Keys (K), Values (V)

To compute self-attention, each input token embedding is projected into three vectors:

1.  **Query (Q):** Represents the current token for which we are computing attention. It "queries" other tokens.
2.  **Key (K):** Represents all tokens in the sequence. The query vector is matched against all key vectors to determine attention scores.
3.  **Value (V):** Also represents all tokens in the sequence. These are the actual values that get aggregated based on the attention scores. If a key has a high attention score with the query, its corresponding value will contribute more to the output for the query token.

These Q, K, and V vectors are created by multiplying the input embeddings (or the output of the previous layer) by learned weight matrices (`W_Q`, `W_K`, `W_V`).

#### Scaled Dot-Product Attention

The most common type of self-attention used in Transformers is Scaled Dot-Product Attention.

*   **Formula:**
    `Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V`

*   **Explanation of Steps:**
    1.  **Dot Product of Queries and Keys (`Q * K^T`):**
        *   Compute the dot product between the query vector of the current token and the key vectors of all tokens in the sequence. This gives a score of how much each token (represented by its key) "matches" or is relevant to the current token (represented by its query).
        *   If `Q` is a matrix of query vectors (one per token) and `K^T` is the transpose of the matrix of key vectors, the result is a matrix of attention scores. `Score_ij` indicates the attention from token `i` (query) to token `j` (key).
    2.  **Scaling (`/ sqrt(d_k)`):**
        *   The attention scores are scaled down by dividing by the square root of the dimension of the key vectors (`d_k`).
        *   **Purpose:** This scaling helps to prevent the dot products from becoming too large, which could push the softmax function into regions where its gradients are very small (making learning difficult).
    3.  **Softmax (`softmax(...)`):**
        *   A softmax function is applied to the scaled attention scores row-wise (for each query token). This converts the scores into probabilities, ensuring they sum to 1.
        *   The output of the softmax, `α_ij`, represents the weight or importance of token `j`'s value for token `i`.
    4.  **Weighted Sum of Values (`... * V`):**
        *   The softmax weights are then multiplied by the value vectors `V`. This gives a weighted sum of the value vectors, where tokens with higher attention scores contribute more to the output.
        *   The result is an output vector for each input token, which is a contextualized representation incorporating information from the entire sequence based on attention.

*   **How Attention Allows Weighing Importance:**
    The softmax scores directly represent the "attention" or weight given to each token's value when computing the representation for the current token. If a token is highly relevant to the current token (high dot product score), it gets a high softmax weight, and its value vector significantly influences the output.

### 2. Multi-Head Attention

Instead of performing a single attention function, Transformers use **Multi-Head Attention**. This allows the model to jointly attend to information from different representation subspaces at different positions.

*   **Operation:**
    1.  **Linear Projections:** The input Q, K, and V vectors are first linearly projected `h` times using different, learned weight matrices for each "head."
        `head_i_Q = Q * W_i^Q`
        `head_i_K = K * W_i^K`
        `head_i_V = V * W_i^V`
        (This means for each head `i` from 1 to `h`, we get a different Q, K, V set).
    2.  **Parallel Attention:** Scaled Dot-Product Attention is then performed in parallel for each of these `h` projected versions of Q, K, V, yielding `h` output vectors (`head_i`).
        `head_i = Attention(head_i_Q, head_i_K, head_i_V)`
    3.  **Concatenation:** The `h` output vectors (`head_1, head_2, ..., head_h`) are concatenated.
    4.  **Final Linear Projection:** The concatenated vector is then projected once more through another learned weight matrix `W_O` to produce the final output of the Multi-Head Attention layer.
        `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O`

*   **Benefit:** Each attention head can learn to focus on different aspects or relationships in the sequence. For example, one head might focus on syntactic relationships, while another focuses on semantic similarities. This provides a richer and more nuanced representation.

### 3. Positional Encoding

Since the Transformer architecture contains no recurrence or convolution, it has no inherent notion of the order or position of tokens in a sequence. Self-attention itself is permutation-invariant. To address this, explicit positional information is added to the input embeddings.

*   **Purpose:** To inject information about the relative or absolute position of tokens in the sequence.
*   **Methods:**
    1.  **Sine and Cosine Functions (Original Transformer):**
        *   Fixed positional encodings are generated using sine and cosine functions of different frequencies:
            `PE(pos, 2i) = sin(pos / 10000^(2i / d_model))`
            `PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))`
            Where:
            *   `pos` is the position of the token in the sequence.
            *   `i` is the dimension index within the embedding.
            *   `d_model` is the dimension of the input embeddings.
        *   These positional encodings are added to the input token embeddings.
        *   **Advantage:** Can generalize to sequences longer than those seen during training. Allows the model to easily learn to attend by relative positions, since `PE(pos+k)` can be represented as a linear function of `PE(pos)`.
    2.  **Learned Positional Embeddings:**
        *   Alternatively, positional embeddings can be learned parameters, similar to word embeddings. A separate embedding vector is learned for each position up to a maximum sequence length.
        *   This is simpler to implement and often works as well or better for many tasks, especially if the sequence lengths are not extremely variable or long. (Used in BERT, GPT).

### 4. Feed-Forward Networks (Position-wise FFN)

Each layer in the Transformer (both encoder and decoder) contains a fully connected feed-forward network (FFN) which is applied to each position separately and identically.

*   **Structure:** Consists of two linear transformations with a ReLU activation function in between.
    `FFN(x) = max(0, x * W_1 + b_1) * W_2 + b_2`
    Where `x` is the output from the attention sub-layer for a specific position.
*   **Purpose:** This FFN adds further non-linearity and processing to each token's representation after the attention mechanism has aggregated information from other tokens. It can be thought of as further transforming or refining the contextualized representation.
*   **Position-wise:** While the same FFN weights (`W_1, b_1, W_2, b_2`) are used for all positions in a sequence within a given layer, the FFN processes each position independently.

### 5. Add & Norm (Layer Normalization and Residual Connections)

Each sub-layer (Self-Attention or FFN) in both the encoder and decoder has a residual connection around it, followed by layer normalization.

*   **Residual Connection:**
    *   The output of the sub-layer is `Sublayer(x)`. The output of the residual connection is `x + Sublayer(x)`.
    *   **Purpose:** Helps to mitigate the vanishing gradient problem in deep networks, allowing for easier training of deeper Transformers. It provides a "shortcut" for gradients and information to flow.
*   **Layer Normalization (LayerNorm):**
    *   Applied after the residual connection.
    *   Normalizes the activations of each layer across the features for a given token, independently for each token in the sequence.
    *   **Purpose:** Stabilizes the learning process by reducing internal covariate shift and smoothing the optimization landscape. It helps to ensure that the inputs to the next layer are well-behaved.
*   **Overall Operation for a Sub-layer:**
    `output = LayerNorm(x + Sublayer(x))`

## The Transformer Encoder

The encoder's role is to map an input sequence of symbol representations (e.g., word embeddings + positional encodings) to a sequence of continuous representations that capture contextual information.

*   **Structure:** A stack of `N` identical layers (e.g., `N=6` in the original paper).
*   **Each Encoder Layer has two sub-layers:**
    1.  **Multi-Head Self-Attention Mechanism:** Computes self-attention on the input sequence (or the output of the previous encoder layer).
    2.  **Position-wise Fully Connected Feed-Forward Network:** Processes each position's representation independently.
*   Each sub-layer is followed by `Add & Norm` (residual connection + layer normalization).

```
Textual Diagram: Single Encoder Layer

Input from previous layer (or input embeddings)
    |
    V
[Multi-Head Self-Attention] ----> Output_Att
    | (residual)                  |
    +-----------------------------+ (add)
    |
    V
[Layer Normalization] ---------> Normed_Output_Att
    |
    V
[Feed-Forward Network] ------> Output_FFN
    | (residual)                  |
    +-----------------------------+ (add)
    |
    V
[Layer Normalization] ---------> Output of Encoder Layer
```

The output of the final encoder layer is a sequence of context-rich embeddings, typically used by the decoder (as K and V vectors) or for downstream tasks in encoder-only models.

## The Transformer Decoder

The decoder's role is to generate an output sequence token by token, typically based on the encoder's output and the previously generated tokens.

*   **Structure:** A stack of `N` identical layers.
*   **Each Decoder Layer has three main sub-layers:**
    1.  **Masked Multi-Head Self-Attention Mechanism:**
        *   Computes self-attention on the output sequence generated so far.
        *   **Masking:** Crucially, a look-ahead mask is applied to the attention scores. This mask prevents positions from attending to subsequent positions. During training, when predicting the token at position `i`, the decoder should only attend to tokens at positions less than `i`. This ensures that the prediction for position `i` can only depend on the known outputs at positions `< i`, maintaining the auto-regressive property.
        *   This is followed by `Add & Norm`.
    2.  **Multi-Head Attention (Encoder-Decoder Attention):**
        *   This layer performs attention over the output of the encoder stack.
        *   **Queries (Q):** Come from the output of the previous sub-layer in the decoder (the masked self-attention output).
        *   **Keys (K) and Values (V):** Come from the output of the encoder stack.
        *   This allows every position in the decoder to attend to all positions in the input sequence, which is crucial for tasks like machine translation where alignment between source and target words is important.
        *   Followed by `Add & Norm`.
    3.  **Position-wise Fully Connected Feed-Forward Network:**
        *   Same structure as in the encoder. Processes each position's representation independently.
        *   Followed by `Add & Norm`.

```
Textual Diagram: Single Decoder Layer

Input from previous decoder layer (or target embeddings, shifted right)
    |
    V
[Masked Multi-Head Self-Attention] ----> Output_Self_Att
    | (residual)                           |
    +--------------------------------------+ (add)
    |
    V
[Layer Normalization] ------------------> Normed_Output_Self_Att
    |                                         | (This becomes Q for next sub-layer)
    |                                         |
    |   Encoder Output (K, V from Encoder Stack)
    |      /     \
    |     /       \
    V    V         V
[Multi-Head Encoder-Decoder Attention] -> Output_EncDec_Att
    | (residual)                               |
    +------------------------------------------+ (add)
    |
    V
[Layer Normalization] ----------------------> Normed_Output_EncDec_Att
    |
    V
[Feed-Forward Network] --------------------> Output_FFN
    | (residual)                               |
    +------------------------------------------+ (add)
    |
    V
[Layer Normalization] -----------------------> Output of Decoder Layer
```

After the final decoder layer, there's typically a linear layer followed by a softmax layer to produce probability distributions over the vocabulary for the next token prediction.

## Training Transformers

*   **Loss Functions:**
    *   For language modeling or machine translation, the typical loss function is **cross-entropy** between the predicted token probabilities and the actual target tokens (one-hot encoded).
*   **Optimization:**
    *   **Adam optimizer** is commonly used.
    *   The original Transformer paper used a specific **learning rate schedule** where the learning rate first increases linearly for a certain number of "warmup steps" and then decreases proportionally to the inverse square root of the step number. This warmup phase helps stabilize training early on.
*   **Regularization:**
    *   **Dropout:** Applied to the outputs of each sub-layer before they are added to the sub-layer input (residual connection) and before the layer normalization. Also applied to the sums of the embeddings and the positional encodings.
    *   **Label Smoothing:** A technique that slightly modifies the target labels (e.g., instead of 1 for the true class and 0 for others, it might be 0.9 for true class and small values for others). This can prevent the model from becoming too confident and improve generalization.

## Popular Transformer Models

### BERT (Bidirectional Encoder Representations from Transformers)

*   **Architecture:** Encoder-only.
*   **Pre-training:** Pre-trained on a massive amount of text data using two unsupervised tasks:
    1.  **Masked Language Model (MLM):** Some percentage of input tokens are randomly masked, and the model tries to predict the original masked tokens based on the context of unmasked tokens. This allows BERT to learn deep bidirectional representations.
    2.  **Next Sentence Prediction (NSP):** The model receives pairs of sentences and predicts whether the second sentence is the actual next sentence that follows the first in the original text. (NSP's utility has been debated in later models).
*   **Fine-tuning:** After pre-training, BERT can be fine-tuned on smaller, task-specific labeled datasets for various downstream tasks (classification, question answering, etc.) by adding a small task-specific output layer.

### GPT (Generative Pre-trained Transformer) Series

*   **Architecture:** Decoder-only.
*   **Pre-training:** Pre-trained on a massive amount of text data using a standard language modeling objective: predict the next token in a sequence given the previous tokens. This is inherently auto-regressive.
*   **Fine-tuning/Prompting:**
    *   Early GPT models were fine-tuned for downstream tasks.
    *   Later models (GPT-2, GPT-3, GPT-4) are so large and powerful that they can perform many tasks in a **zero-shot** or **few-shot** setting by just being prompted with a natural language description or a few examples of the task, without any gradient updates.
*   Known for their impressive text generation capabilities.

### T5 (Text-to-Text Transfer Transformer)

*   **Architecture:** Encoder-Decoder.
*   **Approach:** Frames all NLP tasks as a "text-to-text" problem. The model takes text as input and produces text as output. For example, for translation, input is "translate English to German: That is good." and output is "Das ist gut." For classification, input might be "sentiment: This movie is great!" and output is "positive."
*   Pre-trained on a mixture of unsupervised and supervised tasks.

### ViT (Vision Transformer)

*   **Application:** Applies the Transformer architecture directly to image classification.
*   **Approach:**
    1.  Splits an image into a sequence of fixed-size patches.
    2.  Linearly embeds each patch.
    3.  Adds positional embeddings to these patch embeddings.
    4.  Feeds the resulting sequence of vectors to a standard Transformer encoder.
    5.  A classification head is attached to the output of a special `[CLS]` token (similar to BERT) to perform image classification.
*   Demonstrated that Transformers can achieve state-of-the-art results in computer vision, challenging the dominance of CNNs.

## Advantages of Transformers

1.  **Parallelizability:** Can process all tokens in a sequence simultaneously, leading to faster training and inference compared to RNNs.
2.  **Handling Long-Range Dependencies:** Self-attention allows direct modeling of relationships between any two tokens in a sequence, regardless of their distance, making them very effective at capturing long-range context.
3.  **Scalability:** Transformers have scaled remarkably well to very large models and datasets, leading to significant performance improvements (e.g., GPT-3, BigGAN based on Transformers).
4.  **Transfer Learning Prowess:** Pre-trained Transformer models (like BERT, GPT) have become foundational in NLP, allowing for excellent performance on downstream tasks with relatively little task-specific data.
5.  **Versatility:** Successfully applied to diverse domains beyond NLP, including vision, speech, and reinforcement learning.

## Disadvantages of Transformers

1.  **Computational Cost for Long Sequences:**
    *   The self-attention mechanism has a computational complexity of `O(n^2 * d)` where `n` is the sequence length and `d` is the representation dimension. This quadratic dependency on sequence length makes it very computationally and memory intensive for extremely long sequences.
    *   Research into more efficient attention mechanisms (e.g., Sparse Transformers, Longformers, Linformers) aims to mitigate this.
2.  **Large Data Requirement:** Transformers, especially large ones, typically require very large datasets for pre-training to achieve their full potential.
3.  **Lack of Inherent Positional Information:** Requires explicit positional encodings since the architecture itself doesn't capture sequence order.
4.  **Interpretability:** While attention weights can offer some insights into what the model is focusing on, the internal workings of large Transformers can still be very complex and difficult to interpret fully.
5.  **High Parameter Count:** State-of-the-art Transformer models can have billions or even trillions of parameters, making them expensive to train and deploy.

Transformers represent a paradigm shift in sequence modeling and beyond, with their attention mechanism providing a powerful way to learn contextual representations. Ongoing research continues to address their limitations and expand their capabilities.

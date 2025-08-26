# Data Flow in LLMs

## How does Data Flow through an LLM?

The data flow in an LLM can be divided into two main stages:

*   **Training:** In this stage, the model is trained on a massive amount of text data. The data flows through the model as follows:
    1.  The text data is first converted into a sequence of embeddings.
    2.  The embeddings are then fed into the encoder of the Transformer architecture.
    3.  The encoder outputs a sequence of contextualized embeddings.
    4.  The contextualized embeddings are then fed into the decoder of the Transformer architecture.
    5.  The decoder outputs a sequence of probabilities, which represent the probability of each word in the vocabulary being the next word in the sequence.
    6.  The model is then trained to maximize the probability of the correct next word in the sequence.

*   **Inference:** In this stage, the model is used to generate text. The data flows through the model as follows:
    1.  The user provides a prompt to the model.
    2.  The prompt is first converted into a sequence of embeddings.
    3.  The embeddings are then fed into the encoder of the Transformer architecture.
    4.  The encoder outputs a sequence of contextualized embeddings.
    5.  The contextualized embeddings are then fed into the decoder of the Transformer architecture.
    6.  The decoder outputs a sequence of probabilities, which represent the probability of each word in the vocabulary being the next word in the sequence.
    7.  The model then samples from the probability distribution to generate the next word in the sequence.
    8.  The process is then repeated, with the newly generated word being added to the input sequence, until the model generates an end-of-sequence token.

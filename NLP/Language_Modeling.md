# Language Modeling in NLP

## What is a Language Model (LM)?

### Definition:
A Language Model (LM) is a fundamental concept in Natural Language Processing. It is a **probability distribution over sequences of words (or other linguistic units like characters or subwords)**. Given a sequence of words, an LM assigns a probability to that entire sequence.

More commonly, an LM is used to predict the probability of the next word in a sequence, given the words that have come before it.
`P(w_n | w_1, w_2, ..., w_{n-1})`
where `w_n` is the next word and `w_1, ..., w_{n-1}` are the preceding words.

### Goal:
The primary goal of a language model is to **predict the next word (or character) given a sequence of previous words (or characters)**. Essentially, it learns the likelihood of word sequences occurring in a language. A good LM will assign a higher probability to grammatically correct and semantically plausible sentences than to nonsensical or incorrect ones.

For example, a good LM would assign:
- `P("the cat sat on the mat")` > `P("mat the on cat the sat")`
- `P("I am going to the store")` > `P("I am going to the moon")` (depending on context, but generally)

### Importance in NLP:
Language models are a core component in a vast array of NLP applications because they provide a way to quantify the likelihood of text. Their ability to "understand" and predict sequences of language makes them invaluable for:
-   **Speech Recognition:** Distinguishing between homophones (e.g., "recognize speech" vs. "wreck a nice beach") by choosing the more probable word sequence.
-   **Machine Translation:** Ensuring the output translation is fluent and natural in the target language.
-   **Text Generation:** Creating human-like text for stories, dialogue, code, etc.
-   **Spelling Correction:** Suggesting corrections based on the probability of word sequences.
-   **Autocomplete/Predictive Text:** Suggesting the next word as a user types.
-   **Summarization:** Generating summaries that are coherent and readable.
-   **Question Answering:** Generating fluent and relevant answers.
-   **Information Retrieval:** Ranking documents based on the likelihood of query terms appearing in a relevant context.

## Statistical Language Models

### N-gram Language Models
N-gram models are traditional statistical LMs that make a simplifying assumption about how words depend on each other.

-   **Concept:** Based on the **Markov assumption**. This assumption states that the probability of a word depends only on the `n-1` preceding words.
    -   If `n=1` (Unigram model): `P(w_i)` - Probability of a word is independent of any previous words.
    -   If `n=2` (Bigram model): `P(w_i | w_{i-1})` - Probability of a word depends only on the immediately preceding word.
    -   If `n=3` (Trigram model): `P(w_i | w_{i-2}, w_{i-1})` - Probability of a word depends on the two preceding words.
    -   And so on for higher-order N-grams.
-   **Calculating N-gram Probabilities:**
    -   **Maximum Likelihood Estimation (MLE):** The probabilities are typically estimated from a large text corpus by counting occurrences.
        -   For a bigram model:
            `P(w_i | w_{i-1}) = Count(w_{i-1}, w_i) / Count(w_{i-1})`
            Where `Count(w_{i-1}, w_i)` is the number of times the sequence `w_{i-1}, w_i` appears in the corpus, and `Count(w_{i-1})` is the number of times `w_{i-1}` appears.
        -   For a trigram model:
            `P(w_i | w_{i-2}, w_{i-1}) = Count(w_{i-2}, w_{i-1}, w_i) / Count(w_{i-2}, w_{i-1})`
    -   **Example (Bigram):**
        Corpus: "<s> I am Sam </s> <s> Sam I am </s> <s> I do not like green eggs and ham </s>"
        (<s> and </s> are special start and end of sentence tokens)
        -   `P("am" | "I") = Count("I am") / Count("I") = 2 / 3`
        -   `P("Sam" | "am") = Count("am Sam") / Count("am") = 1 / 2`
        The probability of a sentence `W = w_1, w_2, ..., w_k` is calculated as:
        `P(W) = P(w_1 | <s>) * P(w_2 | w_1) * ... * P(w_k | w_{k-1}) * P(</s> | w_k)` (for a bigram model, assuming appropriate start/end tokens).

-   **Challenges:**
    -   **Sparsity / Zero-Probability Problem:** If an N-gram (a specific sequence of N words) has never appeared in the training corpus, its count will be zero, leading to a zero probability. This is problematic because unseen N-grams are common, especially for larger N. A single zero probability for an N-gram in a sequence will make the probability of the entire sequence zero.
    -   **Storage:** The number of possible N-grams grows exponentially with N and the vocabulary size. Storing counts for all possible N-grams (especially trigrams and above) can require significant memory.

-   **Smoothing Techniques (to address sparsity):**
    Smoothing techniques adjust the MLE probabilities by taking some probability mass from seen N-grams and distributing it to unseen N-grams.
    -   **Laplace (Add-One) Smoothing:**
        -   **Concept:** Add one to all N-gram counts before normalizing. This ensures no N-gram has a zero probability.
        -   **Formula (for bigrams):**
            `P(w_i | w_{i-1}) = (Count(w_{i-1}, w_i) + 1) / (Count(w_{i-1}) + V)`
            Where `V` is the vocabulary size (the number of unique words).
        -   It often overestimates the probability of unseen N-grams and can perform poorly in practice.
    -   **Add-k Smoothing (Lidstone Smoothing):**
        -   **Concept:** A generalization of Laplace smoothing where a small fractional count `k` (e.g., 0.1, 0.01) is added instead of 1.
        -   **Formula (for bigrams):**
            `P(w_i | w_{i-1}) = (Count(w_{i-1}, w_i) + k) / (Count(w_{i-1}) + kV)`
        -   Performance depends on the choice of `k`, often determined via a held-out set.
    -   **Good-Turing Smoothing:**
        -   **Brief Idea:** Uses the count of N-grams that occurred `r+1` times to estimate the probability mass for N-grams that occurred `r` times, particularly for unseen N-grams (those with `r=0`). It's more sophisticated than Add-k.
    -   **Kneser-Ney Smoothing:**
        -   **Brief Idea:** Often considered the state-of-the-art for traditional N-gram models. It's based on the probability of a word appearing as a novel continuation for various preceding words. It incorporates information from lower-order N-gram distributions in a more principled way than simple interpolation. It handles low-frequency N-grams well.

-   **Advantages of N-gram LMs:**
    -   **Simplicity:** Relatively easy to understand and implement.
    -   **Efficiency:** Probabilities can be pre-calculated and stored, making them fast for lookups during tasks like speech recognition.
    -   Can be effective for tasks where local word context is sufficient.

-   **Disadvantages of N-gram LMs:**
    -   **Inability to Capture Long-Range Dependencies:** The Markov assumption limits the context to the previous `n-1` words. They struggle to model dependencies between words that are far apart in a sentence (e.g., "The man who I saw earlier ... **is** here" vs. "The men who I saw earlier ... **are** here").
    -   **Sparsity:** Even with smoothing, sparsity remains a significant issue for higher-order N-grams.
    -   **Storage:** Can be very large for higher N values and large vocabularies.

## Evaluating Language Models

Evaluating the quality of a language model is crucial.

### Perplexity
-   **Definition:** Perplexity (PP or PPL) is the standard intrinsic evaluation metric for language models. It measures how well a probability model predicts a sample (a test set). A lower perplexity score indicates that the language model is better at predicting the sample.
-   **Formula:** Perplexity is the exponentiated average negative log-likelihood of the test set. For a test set `W = w_1, w_2, ..., w_N` (where N is the total number of words in the test set):
    `PP(W) = P(w_1, w_2, ..., w_N)^{-1/N}`
    `PP(W) = exp( -(1/N) * Σ log P(w_i | w_1, ..., w_{i-1}) )`
    Or, more practically, using the chain rule and log probabilities:
    `PP(W) = exp( -(1/N) * log P(W) )`
    where `log P(W) = Σ_{i=1}^{N} log P(w_i | w_{<i})` (log probability of the test set according to the model).
-   **Intuition:** Perplexity can be thought of as the (geometric) average number of choices the language model "thinks" it has when trying to predict the next word. If perplexity is 100, it means that on average, the model is as confused as if it had to choose uniformly and independently among 100 words at each step.
-   **How to calculate it:**
    1.  Train your language model on a training corpus.
    2.  Take a separate test corpus (unseen during training).
    3.  Calculate the probability of the entire test corpus according to your language model using the chain rule.
    4.  Apply the perplexity formula.
-   **Intrinsic vs. Extrinsic Evaluation:**
    -   **Intrinsic Evaluation (like perplexity):** Measures the quality of the model directly on its primary task (predicting text) without regard to a specific downstream application. It's faster and helps in model development.
    -   **Extrinsic Evaluation:** Measures the model's performance on a downstream task (e.g., Word Error Rate in speech recognition, BLEU score in machine translation). This is often the ultimate measure of a model's usefulness but can be slower and more complex to set up.

### Other Metrics (Brief Mention)
-   **Word Error Rate (WER):** Commonly used in Automatic Speech Recognition (ASR) to measure the number of errors (substitutions, deletions, insertions) in the transcribed text compared to a reference transcript.
-   **BLEU Score:** Used in machine translation to compare a candidate translation against one or more reference translations.

## Neural Language Models (Brief Introduction)

While N-gram models were dominant for many years, **Neural Language Models (NLMs)** have largely surpassed them in performance for most tasks. This is primarily due to their ability to:
-   **Handle Long-Range Dependencies:** Architectures like Recurrent Neural Networks (RNNs), LSTMs, GRUs, and especially Transformers can capture relationships between words that are far apart in a sequence.
-   **Learn Distributed Representations:** Neural models learn dense vector representations (embeddings) for words, allowing them to capture semantic similarities between words (e.g., "cat" and "feline" might have similar representations). This helps with generalization and combating sparsity, as the model doesn't just rely on exact word matches.

*(Details on RNNs, LSTMs, GRUs will be covered in `Sequence_Models.md`, and Transformers will be detailed in `Transformers.md`.)*

## Applications of Language Models

Language models are foundational to many NLP applications:
-   **Autocomplete / Predictive Text Input:** Suggesting the next word or phrase as a user types (e.g., on smartphones, in search engines).
-   **Spelling Correction:** Identifying and correcting misspelled words by choosing more probable sequences (e.g., "I went to the librari" -> "I went to the library").
-   **Speech Recognition (ASR):** Converting spoken audio into text. LMs help the ASR system choose between acoustically similar word sequences by favoring more linguistically plausible ones.
-   **Machine Translation (MT):** Ensuring the translated output is fluent and grammatically correct in the target language. The LM for the target language scores candidate translations.
-   **Text Generation:** Creating coherent and contextually relevant text for various purposes, including creative writing, chatbots, and code generation.
-   **Optical Character Recognition (OCR):** Improving the accuracy of recognized text by selecting more probable word sequences.
-   **Information Retrieval:** Ranking search results based on the likelihood that a document contains information relevant to a query.
-   **Summarization:** Generating fluent and coherent summaries.

## Libraries for N-gram Modeling

-   **NLTK (Natural Language Toolkit):**
    -   Provides tools for creating and evaluating N-gram models, including functions for padding sentences, generating N-grams, and basic smoothing.
    -   Good for educational purposes and smaller-scale experiments.
    -   `nltk.lm` module.
-   **SRILM (SRI Language Modeling Toolkit):**
    -   An external, command-line toolkit widely used in the research community for building and evaluating N-gram language models.
    -   Supports various smoothing techniques (including Kneser-Ney) and is optimized for large-scale models.
-   **KenLM:**
    -   Another popular toolkit, known for its efficiency in terms of memory and speed, especially for large N-gram models.

While neural network libraries (PyTorch, TensorFlow/Keras) are used for Neural Language Models, the above are more specific to traditional N-gram approaches.
```

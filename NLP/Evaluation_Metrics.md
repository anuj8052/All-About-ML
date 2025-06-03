# Evaluation Metrics in NLP

## Introduction to Evaluation in NLP

### Why Evaluation is Crucial:
Evaluation is a fundamental aspect of developing and understanding NLP models. It allows us to:
-   **Quantify Performance:** Objectively measure how well a model performs on a specific task.
-   **Compare Models:** Compare different models or variations of the same model to identify the most effective approaches.
-   **Guide Development:** Identify weaknesses in a model and areas for improvement.
-   **Ensure Reliability:** Assess whether a model is suitable for deployment in real-world applications.
-   **Benchmark Progress:** Track progress in the field by comparing against established benchmarks.

### Intrinsic vs. Extrinsic Evaluation:
-   **Intrinsic Evaluation:** Measures the quality of a model directly on a specific NLP subtask, often in isolation. For example, evaluating a language model using perplexity or a POS tagger using accuracy.
    -   **Pros:** Easier and faster to compute, helps in debugging and iterative model improvement.
    -   **Cons:** May not directly correlate with performance on a real-world downstream application.
-   **Extrinsic Evaluation (Task-based Evaluation):** Measures the performance of a model on a downstream application or a more complex task. For example, evaluating the impact of a new NER model on the performance of a question answering system.
    -   **Pros:** Provides a more realistic measure of the model's utility.
    -   **Cons:** Can be more complex, time-consuming, and expensive to set up and run.

### Baselines and SOTA (State-of-the-Art):
-   **Baselines:** Simpler models or existing standard methods that provide a point of comparison. A new model should ideally perform better than relevant baselines to demonstrate its value.
-   **State-of-the-Art (SOTA):** The best-performing models or techniques currently known for a particular task on a specific benchmark dataset. Researchers often strive to achieve or surpass SOTA.

---

## Metrics for Classification Tasks
(e.g., Text Classification, Sentiment Analysis, NLI, Spam Detection)

These tasks involve assigning a predefined category/label to an input. Let TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.

### Accuracy:
-   **Definition:** The proportion of correctly classified instances out of the total number of instances.
-   **Formula:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
-   **Pros:** Simple to understand and compute.
-   **Cons:** Can be misleading for imbalanced datasets, where one class is much more frequent than others. A model predicting the majority class all the time might have high accuracy but be useless.

### Precision (Positive Predictive Value):
-   **Definition:** The proportion of correctly predicted positive instances out of all instances predicted as positive. Answers: "Of all instances predicted as positive, how many were actually positive?"
-   **Formula:** `Precision = TP / (TP + FP)`
-   **Relevance:** High precision is important when the cost of a False Positive is high.
    -   *Example:* In spam detection, high precision means that when an email is classified as spam, it is very likely to actually be spam (minimizing legitimate emails going to spam).

### Recall (Sensitivity, True Positive Rate - TPR):
-   **Definition:** The proportion of correctly predicted positive instances out of all actual positive instances. Answers: "Of all actual positive instances, how many did the model correctly identify?"
-   **Formula:** `Recall = TP / (TP + FN)`
-   **Relevance:** High recall is important when the cost of a False Negative is high.
    -   *Example:* In medical diagnosis for a serious disease, high recall means that the model correctly identifies most patients who actually have the disease (minimizing missed diagnoses).

### F1-Score:
-   **Definition:** The harmonic mean of Precision and Recall. It provides a single score that balances both metrics.
-   **Formula:** `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`
-   **Relevance:** Useful when you want a balance between Precision and Recall, especially when dealing with imbalanced classes. It penalizes models that are extremely good at one metric but poor at the other.

### Confusion Matrix:
-   **Definition:** A table used to visualize the performance of a classification model. Rows typically represent the actual classes, and columns represent the predicted classes (or vice-versa).
-   **Structure:**
    ```
              Predicted: Class A  |  Predicted: Class B
    Actual: Class A |     TP            |       FN
    Actual: Class B |     FP            |       TN
    ```
    (for a binary classification problem)
-   **Benefit:** Provides a detailed breakdown of correct and incorrect classifications for each class, helping to identify specific error patterns.

### Area Under the ROC Curve (AUC-ROC):
-   **ROC Curve (Receiver Operating Characteristic Curve):** A plot of the True Positive Rate (Recall) against the False Positive Rate (FPR) at various classification thresholds.
    -   `FPR = FP / (FP + TN)`
-   **AUC (Area Under the Curve):** Represents the degree or measure of separability between classes. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0. An AUC of 0.5 suggests no discrimination (like random guessing).
-   **Benefit:** AUC-ROC is threshold-independent and useful for evaluating models across the entire range of their decision thresholds. Good for imbalanced datasets.

### Log Loss (Cross-Entropy Loss):
-   **Definition:** Measures the performance of a classification model where the prediction input is a probability value between 0 and 1. Log loss increases as the predicted probability diverges from the actual label.
-   **Formula (for binary classification):** `Log Loss = - (1/N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]`
    -   `y_i` is the actual label (0 or 1).
    -   `p_i` is the predicted probability of class 1.
-   **Benefit:** Penalizes confident but incorrect predictions more heavily.

### Macro vs. Micro Averaging for Multi-class Classification:
When dealing with multi-class classification, precision, recall, and F1-score can be averaged across classes:
-   **Macro-Averaging:** Calculate the metric independently for each class and then take the unweighted average. Treats all classes equally, regardless of their size.
    -   `Macro-Precision = (Precision_Class1 + ... + Precision_ClassK) / K`
-   **Micro-Averaging:** Aggregate the contributions of all classes to compute the average metric. Sum up individual TPs, FPs, FNs for each class first. Gives equal weight to each instance, so larger classes have more influence.
    -   `Micro-Precision = (TP_Class1 + ... + TP_ClassK) / ((TP_Class1 + ... + TP_ClassK) + (FP_Class1 + ... + FP_ClassK))`
    (Micro-Recall and Micro-F1 are calculated similarly and Micro-F1 = Micro-Precision = Micro-Recall = Accuracy if each instance has exactly one label).
-   **Weighted-Averaging:** Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).

---

## Metrics for Sequence Labeling Tasks
(e.g., Named Entity Recognition - NER, Part-of-Speech - POS Tagging)

These tasks involve assigning a label to each token in a sequence.
-   Often use standard classification metrics like **Precision, Recall, and F1-score** applied at the token level or, more commonly, at the entity/chunk level.
-   **For POS Tagging:** Accuracy (percentage of correctly tagged tokens) is common.
-   **For NER:**
    -   Metrics are typically calculated based on correctly identified entity spans and their types.
    -   **CoNLL F1-Score:** The standard evaluation metric for NER competitions (like CoNLL-2003). It requires an exact match of both the entity boundaries (span) and the entity type. Partial matches are counted as incorrect.

---

## Metrics for Machine Translation (MT)

### BLEU (Bilingual Evaluation Understudy):
-   **Definition:** Measures the n-gram precision overlap between the machine-generated translation and one or more human reference translations. It also includes a brevity penalty (BP) to penalize translations that are too short.
-   **Formula Idea:** `BLEU = BP * exp( Σ w_n * log(p_n) )`
    -   `p_n` is the modified n-gram precision for n-grams of length up to N (typically N=4).
    -   `w_n` are weights (usually uniform 1/N).
-   **Pros:** Fast to compute, language-independent, widely adopted.
-   **Cons:** Correlates moderately with human judgment, doesn't consider semantic similarity (synonyms, paraphrases), can be insensitive to grammatical errors that don't affect n-gram overlap, favors shorter translations if BP is not effective.

### METEOR (Metric for Evaluation of Translation with Explicit ORdering):
-   **Definition:** Computes a score based on unigram precision and recall, with a focus on alignments between translation and reference. It considers stemming, synonyms (via WordNet), and paraphrasing.
-   **Pros:** Generally correlates better with human judgment of translation quality than BLEU.
-   **Cons:** More complex to compute than BLEU, relies on language-specific resources like WordNet for synonyms.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
-   While primarily for summarization, variants like ROUGE-L can be used in MT. (See Text Summarization section).

### TER (Translation Edit Rate):
-   **Definition:** Measures the minimum number of edits (insertions, deletions, substitutions, shifts) required to change the machine translation output to match a human reference translation, normalized by the length of the reference.
-   **Pros:** Intuitively understandable as an "error rate."
-   **Cons:** Can be labor-intensive if human annotation of edits is required, though automated versions exist.

---

## Metrics for Text Summarization

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
-   **Definition:** A set of metrics that compare an automatically produced summary against one or more human-written reference summaries by counting overlapping units like n-grams, word sequences, and word pairs.
-   **Common Variants:**
    -   **ROUGE-N:** Measures n-gram recall between candidate and reference summaries. (e.g., ROUGE-1 for unigrams, ROUGE-2 for bigrams).
    -   **ROUGE-L:** Measures the Longest Common Subsequence (LCS) between candidate and reference. Favors summaries that preserve sentence-level structure.
    -   **ROUGE-S / ROUGE-SU:** Measures skip-bigram (any pair of words in sentence order, with arbitrary gaps) or skip-bigram plus unigram co-occurrence statistics.
-   **Pros:** Widely used, correlates reasonably well with human judgments of informativeness.
-   **Cons:** Primarily recall-oriented, may not capture fluency or coherence well, sensitive to the choice of reference summaries.

### BERTScore:
-   **Definition:** Leverages pre-trained contextual embeddings (from BERT or similar models) to compute semantic similarity between tokens in the candidate summary and reference summary. It computes precision, recall, and F1-score based on cosine similarity of these embeddings.
-   **Pros:** Better captures semantic similarity than n-gram overlap, shown to correlate well with human judgments.
-   **Cons:** Computationally more intensive than ROUGE, depends on the quality of the underlying pre-trained model.

### Human Evaluation:
-   Often crucial for summarization. Evaluators assess summaries based on criteria like:
    -   **Informativeness/Content Coverage:** Does the summary capture the main points?
    -   **Fluency:** Is the summary grammatically correct and easy to read?
    -   **Coherence:** Do the sentences flow logically?
    -   **Succinctness:** Is the summary concise?
    -   **Factual Consistency (for abstractive):** Does the summary accurately reflect the source?

---

## Metrics for Language Modeling

### Perplexity (PPL):
-   **Definition:** A measure of how well a probability model predicts a sample. Lower perplexity indicates the model is better at predicting the test set. It's the exponentiated average negative log-likelihood of the test set.
-   **(Recap from `Language_Modeling.md`):** `PP(W) = exp( -(1/N) * Σ log P(w_i | w_1, ..., w_{i-1}) )`
-   **Pros:** Standard intrinsic evaluation for language models.
-   **Cons:** Sensitive to vocabulary size, may not directly correlate with performance on downstream tasks. Comparing perplexity scores is only meaningful if vocabularies and tokenization are consistent.

### Bits Per Character (BPC) / Bits Per Word (BPW):
-   **Definition:** Related to perplexity, measures the average number of bits needed to encode each character or word according to the language model.
-   `BPC = log2(Perplexity) / (Avg characters per token)` (approximate relation for character LMs).
-   Lower BPC/BPW is better.

---

## Metrics for Question Answering (QA)

### Exact Match (EM):
-   **Definition:** For extractive QA, this metric measures the percentage of predictions where the predicted answer span matches the ground truth answer span exactly.
-   **Pros:** Simple and strict.
-   **Cons:** Can be too harsh, as a slightly different but semantically correct answer span is penalized.

### F1-Score:
-   **Definition:** For extractive QA, treats the prediction and ground truth answers as bags of tokens and computes the F1-score based on the overlap of these tokens.
-   **Pros:** More lenient than EM, accounts for partial overlaps.
-   **Cons:** Doesn't fully capture semantic correctness if token overlap is misleading.

### Human Evaluation:
-   Often necessary to assess the actual correctness and helpfulness of answers, especially for abstractive or open-domain QA.

---

## Metrics for Text Generation (Open-ended)
(e.g., Story Generation, Dialogue Response Generation)

Evaluating open-ended text generation is notoriously difficult with automatic metrics.
-   **Automatic Metrics (with limitations):**
    -   **Perplexity:** Can measure fluency to some extent.
    -   **BLEU, ROUGE:** Can be used if reference texts are available (e.g., for style transfer or paraphrasing), but often inadequate for creativity or coherence.
    -   **Diversity Metrics:** Measure the variety of n-grams or vocabulary used in the generated text to avoid repetitive output (e.g., distinct-1, distinct-2).
-   **Human Evaluation is Key:** Assess based on:
    -   **Fluency:** Grammatical correctness and readability.
    -   **Coherence:** Logical flow and consistency of ideas.
    -   **Relevance/Context-appropriateness:** How well it fits the prompt or context.
    -   **Informativeness, Engagingness, Creativity, etc.** (depending on the specific task).

---

## Metrics for Dialogue Systems

Evaluation can be complex, combining aspects of NLU, DM, and NLG.
-   **Turn-level metrics:**
    -   **NLU Accuracy:** Intent recognition accuracy, slot filling F1-score.
    -   **Response Quality:** Perplexity, BLEU (if reference responses exist), semantic similarity to references.
-   **Dialogue-level metrics:**
    -   **Task Completion Rate (Success Rate):** For task-oriented systems, whether the user successfully achieved their goal.
    -   **Conversation Length / Number of Turns:** Shorter can be better if the task is completed efficiently.
    -   **Dialogue Coherence.**
-   **User Satisfaction:**
    -   Often measured through surveys (e.g., Likert scales) asking users to rate aspects like helpfulness, ease of use, and overall experience. This is a crucial extrinsic metric.

---

## Importance of Human Evaluation

### Limitations of Automatic Metrics:
-   Many automatic metrics (especially for generation tasks like MT, summarization, dialogue) are based on surface-level overlap (e.g., n-grams) and may not capture semantic meaning, fluency, coherence, or factual correctness adequately.
-   They can sometimes be "gamed" by models that optimize for the metric without producing genuinely good output.
-   Correlation with human judgment can be weak for some metrics and tasks.

### When and How to Conduct Human Evaluation:
-   **When:**
    -   When developing new tasks or metrics.
    -   For a final assessment of system quality before deployment.
    -   When automatic metrics are known to be insufficient for the task.
    -   To compare systems when automatic metrics show small differences.
-   **How:**
    -   Define clear evaluation criteria and guidelines for annotators.
    -   Use multiple annotators to ensure reliability and measure inter-annotator agreement (e.g., using Kappa score).
    -   Collect ratings on scales (e.g., Likert scales for fluency, coherence) or rankings of different system outputs.
    -   Can be expensive and time-consuming.

Ultimately, a combination of automatic and human evaluation is often needed for a comprehensive understanding of an NLP model's performance.
```

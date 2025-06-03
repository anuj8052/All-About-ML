# Common Natural Language Processing (NLP) Tasks

This document provides an overview of common tasks in Natural Language Processing, outlining their definitions, inputs/outputs, methods, evaluation metrics, challenges, and applications.

---

## 1. Text Classification (Categorization)

-   **Definition:** The task of assigning a predefined category or label to a piece of text (document, sentence, query, etc.).
-   **Input:** A piece of text.
-   **Output:** One or more predefined categories/labels for that text.
-   **Common Sub-tasks / Examples:**
    -   **Sentiment Analysis:** Determining the emotional tone (positive, negative, neutral) – *detailed further below*.
    -   **Topic Labeling/Classification:** Identifying the main topic(s) of a text (e.g., sports, politics, technology).
    -   **Spam Detection:** Classifying emails or messages as spam or not spam.
    -   **Intent Recognition:** Identifying the user's intent behind a query or command (e.g., in a chatbot: "book a flight", "check weather").
    -   **Language Identification:** Determining the language of a given text.
-   **Methods/Approaches:**
    -   **Traditional:** Naive Bayes, Support Vector Machines (SVMs) with TF-IDF features, Logistic Regression.
    -   **Deep Learning:** Convolutional Neural Networks (CNNs) for text, Recurrent Neural Networks (LSTMs, GRUs), Transformer-based models (e.g., BERT, RoBERTa, XLNet for fine-tuning).
-   **Evaluation Metrics:**
    -   Accuracy, Precision, Recall, F1-score (especially for imbalanced datasets).
    -   Area Under the ROC Curve (AUC).
-   **Key Challenges:**
    -   Ambiguity in language.
    -   Handling sarcasm, irony.
    -   Domain-specific vocabulary and context.
    -   Class imbalance.
    -   Scalability for a large number of categories.
-   **Real-world Applications:**
    -   Filtering spam emails.
    -   Organizing documents by topic.
    -   Understanding customer feedback (sentiment).
    -   Routing customer support queries based on intent.
    -   Content moderation.

---

## 2. Named Entity Recognition (NER)

-   **Definition:** The task of identifying and categorizing named entities (mentions of real-world objects) in text into predefined categories.
-   **Input:** A sequence of text.
-   **Output:** A sequence of labels identifying named entities and their types.
    -   Common entity types: PER (Person), ORG (Organization), LOC (Location), GPE (Geo-Political Entity), DATE, TIME, MONEY, PRODUCT, EVENT.
    -   Often uses IOB tagging (Inside, Outside, Beginning) or BIOES (Beginning, Inside, Outside, End, Single) format.
-   **Methods/Approaches:**
    -   **Traditional:** Rule-based systems (gazetteers, regular expressions), Conditional Random Fields (CRFs) with handcrafted features.
    -   **Deep Learning:** BiLSTMs with a CRF layer on top, Transformer-based models (e.g., BERT, spaCy's transformers) fine-tuned for token classification.
-   **Evaluation Metrics:**
    -   Precision, Recall, F1-score at the entity level (exact match of span and type).
-   **Key Challenges:**
    -   Ambiguity (e.g., "Washington" can be a person, location, or organization).
    -   Nested entities (e.g., "[Bank of [China]]").
    -   Identifying entities not seen during training (OOV entities).
    -   Consistency across long documents.
    -   Fine-grained NER with many categories.
-   **Real-world Applications:**
    -   Information extraction from news articles, legal documents, medical records.
    -   Powering search engines to understand entities in queries.
    -   Content recommendation.
    -   Customer support (extracting product names, customer IDs).
    -   Knowledge graph population.

---

## 3. Part-of-Speech (POS) Tagging

-   **Definition:** The process of assigning a grammatical category (part-of-speech tag) to each word in a sentence.
-   **Input:** A sequence of words (a sentence).
-   **Output:** A sequence of POS tags, one for each word.
    -   Common POS tags: Noun (NN), Verb (VB), Adjective (JJ), Adverb (RB), Preposition (IN), Pronoun (PRP), Determiner (DT), Conjunction (CC), etc. (Tagsets like Penn Treebank tagset are common).
-   **Methods/Approaches:**
    -   **Traditional:** Rule-based taggers, Hidden Markov Models (HMMs), Maximum Entropy Markov Models (MEMMs), Conditional Random Fields (CRFs).
    -   **Deep Learning:** BiLSTMs, Transformer-based models (BERT, etc.) fine-tuned for token classification.
-   **Evaluation Metrics:**
    -   Accuracy (percentage of correctly tagged words).
-   **Key Challenges:**
    -   Word ambiguity (e.g., "book" can be a noun or a verb; "set" has many meanings and POS tags).
    -   Handling unknown words or new word usages.
    -   Fine-grained tagsets.
-   **Real-world Applications:**
    -   Prerequisite for many downstream NLP tasks like parsing, NER, machine translation, and information extraction.
    -   Text-to-speech systems (for correct pronunciation).
    -   Grammar checking.
    -   Lexicography.

---

## 4. Text Summarization

-   **Definition:** The task of creating a concise and fluent summary of a longer text document that captures its main ideas.
-   **Input:** A source text document (or multiple documents).
-   **Output:** A shorter text representing the summary.
-   **Types:**
    -   **Extractive Summarization:**
        -   **Concept:** Selects important sentences (or phrases) directly from the source document and concatenates them to form a summary.
        -   **Methods:** TF-IDF based scoring, graph-based methods (e.g., TextRank, LexRank), machine learning models to score sentence importance.
    -   **Abstractive Summarization:**
        -   **Concept:** Generates new sentences that paraphrase and condense the information in the source document. This is more human-like but harder.
        -   **Methods:** Sequence-to-sequence (Seq2Seq) models with RNNs (LSTMs/GRUs) and attention, Transformer-based models (e.g., BART, T5, Pegasus, GPT variants).
-   **Evaluation Metrics:**
    -   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Compares the n-gram overlap between the generated summary and reference (human-written) summaries (ROUGE-N, ROUGE-L, ROUGE-SU).
    -   **BLEU (Bilingual Evaluation Understudy):** Though primarily for MT, sometimes adapted.
    -   Human evaluation (fluency, coherence, informativeness).
    -   More recent metrics like BERTScore.
-   **Key Challenges:**
    -   **Abstractive:** Generating factually consistent, coherent, and fluent summaries. Avoiding hallucination (generating information not in the source).
    -   **Extractive:** Sentence redundancy, ensuring coherence between selected sentences.
    -   Maintaining important information and context.
    -   Handling long documents efficiently.
    -   Evaluating summary quality objectively.
-   **Real-world Applications:**
    -   News summarization (e.g., Google News, news apps).
    -   Summarizing scientific articles, legal documents, financial reports.
    -   Generating headlines or snippets for search results.
    -   Meeting summarization.

---

## 5. Machine Translation (MT)

-   **Definition:** The task of automatically translating text from one natural language (source language) to another (target language) while preserving meaning.
-   **Input:** A sentence or document in the source language.
-   **Output:** The corresponding sentence or document in the target language.
-   **Methods/Approaches:**
    -   **Traditional:** Rule-based MT (RBMT), Statistical Machine Translation (SMT - e.g., phrase-based, hierarchical).
    -   **Deep Learning (Neural Machine Translation - NMT):** Sequence-to-sequence models using RNNs (LSTMs/GRUs) with attention mechanisms. Transformer-based models now dominate NMT.
-   **Evaluation Metrics:**
    -   **BLEU (Bilingual Evaluation Understudy):** Measures n-gram precision overlap between machine translation and human reference translations.
    -   **METEOR (Metric for Evaluation of Translation with Explicit ORdering).**
    -   **TER (Translation Edit Rate).**
    -   Human evaluation (fluency, adequacy).
-   **Key Challenges:**
    -   Handling ambiguity (lexical, syntactic).
    -   Translating idioms and cultural nuances.
    -   Maintaining grammatical correctness and fluency in the target language.
    -   Dealing with morphologically rich languages.
    -   Low-resource language pairs (limited parallel training data).
    -   Domain adaptation (e.g., translating casual text vs. technical manuals).
-   **Real-world Applications:**
    -   Online translation services (Google Translate, DeepL, Bing Translator).
    -   Cross-lingual communication tools.
    -   Localizing software and websites.
    -   Assisting human translators.

---

## 6. Question Answering (QA)

-   **Definition:** The task of providing an answer to a question posed in natural language.
-   **Input:** A question and often a context document (or a large corpus) from which to find/generate the answer.
-   **Output:** The answer to the question, which can be a span of text, a generated sentence, or a choice from multiple options.
-   **Types:**
    -   **Extractive QA:**
        -   **Concept:** The answer is a direct span of text extracted from the provided context document.
        -   **Methods:** Models predict the start and end tokens of the answer span within the context (e.g., BERT fine-tuned for SQuAD dataset).
    -   **Abstractive QA:**
        -   **Concept:** The answer is generated (paraphrased or synthesized) based on the information in the context, not necessarily a direct quote.
        -   **Methods:** Sequence-to-sequence models, often Transformer-based (e.g., T5, BART).
    -   **Open-Domain QA:** Answers questions based on a large corpus of documents (e.g., Wikipedia) or the open web. Often involves a retrieval step (finding relevant documents) followed by a reading comprehension (extractive/abstractive) step.
    -   **Closed-Domain QA:** Answers questions related to a specific domain or a limited set of documents (e.g., answering questions about a company's internal knowledge base).
    -   **Yes/No QA:** Answering with a simple "yes" or "no".
    -   **Multiple-Choice QA:** Selecting the correct answer from a list of options.
-   **Evaluation Metrics:**
    -   **Extractive QA:** Exact Match (EM), F1-score (token-level overlap).
    -   **Abstractive QA:** ROUGE, BLEU, human evaluation.
    -   Accuracy for multiple-choice or yes/no.
-   **Key Challenges:**
    -   Understanding the question's intent and nuances.
    -   Requiring reasoning over multiple pieces of information in the context.
    -   Handling questions that require common-sense knowledge or implicit information.
    -   Generating fluent and factually correct answers (for abstractive QA).
    -   Scalability for open-domain QA (efficiently searching vast amounts of text).
-   **Real-world Applications:**
    -   Virtual assistants (Siri, Alexa, Google Assistant).
    -   Search engines providing direct answers.
    -   Customer support chatbots.
    -   Educational tools.

---

## 7. Sentiment Analysis

-   **Definition:** (Also a sub-task of Text Classification) The process of identifying and categorizing the emotional tone or subjective opinion expressed in a piece of text.
-   **Input:** A piece of text (sentence, document, review, tweet).
-   **Output:** A sentiment label.
    -   **Polarity:** Commonly positive, negative, or neutral.
    -   **Fine-grained Polarity:** E.g., very positive, positive, neutral, negative, very negative.
    -   **Aspect-Based Sentiment Analysis (ABSA):** Identifying sentiment towards specific aspects or features of an entity (e.g., "The *camera* on this phone is *great* [positive], but the *battery life* is *terrible* [negative].").
    -   **Emotion Detection:** Identifying specific emotions like joy, anger, sadness, fear.
-   **Methods/Approaches:**
    -   **Lexicon-based:** Using sentiment dictionaries (lists of words with pre-assigned sentiment scores).
    -   **Traditional ML:** Naive Bayes, SVMs with features like n-grams, TF-IDF.
    -   **Deep Learning:** CNNs, RNNs (LSTMs/GRUs), Transformer-based models (BERT, etc.) fine-tuned for classification. For ABSA, more complex attention mechanisms or specialized architectures are often used.
-   **Evaluation Metrics:**
    -   Accuracy, Precision, Recall, F1-score.
    -   Mean Absolute Error (MAE) or Mean Squared Error (MSE) for sentiment intensity scores.
-   **Key Challenges:**
    -   Detecting sarcasm, irony, and figurative language.
    -   Handling negation and its scope (e.g., "not good").
    -   Context dependency (e.g., "small" can be positive for a phone but negative for a hotel room).
    -   Implicit sentiment.
    -   Domain-specific sentiment words.
    -   Aspect extraction for ABSA.
-   **Real-world Applications:**
    -   Monitoring brand reputation and social media sentiment.
    -   Analyzing customer reviews for products and services.
    -   Gauging public opinion on political candidates or policies.
    -   Market research and trend analysis.

---

## 8. Language Modeling

-   **Definition:** The task of predicting the probability of a sequence of words (or other linguistic units). More commonly, it involves predicting the next word in a sequence given the preceding words.
-   **Input:** A sequence of words (prefix).
-   **Output:** A probability distribution over the vocabulary for the next word, or the probability of the entire sequence.
-   **(Note:** This is extensively covered in `Language_Modeling.md`. It's listed here because it's a fundamental NLP task that underpins many other applications and pre-training objectives.)
-   **Methods/Approaches:**
    -   **Statistical:** N-gram models.
    -   **Neural:** RNNs (LSTMs/GRUs), Transformers (GPT, BERT's MLM).
-   **Evaluation Metrics:**
    -   Perplexity, Bits-Per-Character (BPC).
-   **Key Challenges:**
    -   Handling long-range dependencies.
    -   Sparsity (unseen word sequences).
    -   Computational cost for large vocabularies and contexts.
-   **Real-world Applications:**
    -   Autocomplete / Predictive text input.
    -   Speech recognition (choosing between acoustically similar words).
    -   Machine translation (scoring fluency of translations).
    -   Text generation.
    -   Spelling and grammar correction.
    -   Core of pre-training for many large language models.

---

## 9. Text Generation (Natural Language Generation - NLG)

-   **Definition:** The process of producing natural language text from non-linguistic data or based on some input text.
-   **Input:** Can vary widely: structured data (tables, database records), an image, a prompt (a few words or sentences), a knowledge base, or context from a dialogue.
-   **Output:** Coherent, fluent, and contextually relevant natural language text.
-   **Methods/Approaches:**
    -   **Traditional:** Template-based, rule-based systems.
    -   **Deep Learning:**
        -   Language Models (LMs) are inherently generative (e.g., GPT-family models are explicitly trained for this).
        -   Sequence-to-sequence models (RNNs with attention, Transformers) for conditional generation (e.g., translation, summarization, dialogue response).
        -   Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) have also been explored, though less dominant than LMs.
-   **Evaluation Metrics:**
    -   Perplexity (for LM-based generation).
    -   BLEU, ROUGE (if reference texts are available, e.g., for summarization or style transfer).
    -   Human evaluation (fluency, coherence, creativity, task-specific quality).
    -   Diversity metrics (to avoid repetitive text).
-   **Key Challenges:**
    -   Maintaining coherence and consistency over long generated texts.
    -   Controllability (generating text with specific attributes, style, or content).
    -   Avoiding factual inaccuracies or "hallucinations."
    -   Ensuring diversity and avoiding repetitive or generic output.
    -   Ethical concerns (e.g., generation of fake news, harmful content).
-   **Real-world Applications:**
    -   Creative writing (story generation, poetry).
    -   Code generation.
    -   Dialogue systems and chatbots generating responses.
    -   Image captioning.
    -   Automated report generation (e.g., from financial data or weather data).
    -   Personalized content creation.

---

## 10. Coreference Resolution

-   **Definition:** The task of identifying all expressions (mentions) in a text that refer to the same real-world entity.
-   **Input:** A text document.
-   **Output:** Clusters of mentions, where each cluster refers to the same entity.
    -   Example: In "Susan said *she* would be late. *Her* car broke down.", {"Susan", "she", "Her"} form a coreference cluster.
-   **Methods/Approaches:**
    -   **Traditional:** Rule-based (Hobbs algorithm), mention-pair models using features and classifiers.
    -   **Deep Learning:** End-to-end models, often using LSTMs or Transformers to learn mention representations and then scoring potential coreferent pairs or clustering mentions. Span-based models are common.
-   **Evaluation Metrics:**
    -   MUC, B-cubed, CEAF, CoNLL F1 (average of MUC, B-cubed, CEAF).
-   **Key Challenges:**
    -   Distinguishing between different entities with similar names or descriptions.
    -   Handling pronouns and anaphora correctly.
    -   Resolving coreferences across long distances in text.
    -   Incorporating world knowledge (e.g., knowing that "CEO" and the person's name might refer to the same entity).
-   **Real-world Applications:**
    -   Information extraction (to consolidate information about an entity).
    -   Text summarization (to understand entity prominence).
    -   Machine translation (for correct pronoun translation).
    -   Question answering (to link questions to relevant entities in context).
    -   Dialogue systems.

---

## 11. Relation Extraction

-   **Definition:** The task of identifying and categorizing semantic relationships between pairs of named entities in text.
-   **Input:** A text document (often with pre-identified named entities).
-   **Output:** Tuples of the form `(entity1, relationship_type, entity2)`.
    -   Example: In "Apple Inc. is headquartered in Cupertino.", the relation is `(Apple Inc., HeadquarteredIn, Cupertino)`.
-   **Methods/Approaches:**
    -   **Rule-based:** Using dependency parse paths or specific lexical patterns between entities.
    -   **Supervised ML:** Training classifiers on features derived from entity pairs and their context (e.g., SVMs, Logistic Regression).
    -   **Deep Learning:**
        -   CNNs or RNNs over the shortest dependency path or the sentence context between entities.
        -   Transformer-based models fine-tuned by adding markers around entities and classifying the relationship based on representations like the `[CLS]` token or entity start tokens.
    -   **Distant Supervision:** Using existing knowledge bases (e.g., Freebase, Wikidata) to automatically label training data, though this can be noisy.
-   **Evaluation Metrics:**
    -   Precision, Recall, F1-score for each relation type and overall.
-   **Key Challenges:**
    -   Handling complex sentences with multiple entities and relations.
    -   Distinguishing between different relations that might have similar surface patterns.
    -   Dealing with relations expressed implicitly or across sentence boundaries.
    -   Limited labeled data for many specific relation types.
    -   Noise from distant supervision.
-   **Real-world Applications:**
    -   Knowledge base construction and population.
    -   Improving search engine understanding of entities and their connections.
    -   Biomedical NLP (e.g., extracting protein-protein interactions, drug-disease relationships).
    -   Financial analysis (e.g., company-executive relationships).

---

## 12. Natural Language Inference (NLI) / Textual Entailment

-   **Definition:** The task of determining the logical relationship between two text fragments: a "premise" and a "hypothesis."
-   **Input:** A pair of sentences: (premise, hypothesis).
-   **Output:** A label indicating the relationship:
    -   **Entailment:** The hypothesis can be logically inferred from the premise. (e.g., P: "A cat is sleeping on the mat." H: "An animal is on the mat.")
    -   **Contradiction:** The hypothesis logically contradicts the premise. (e.g., P: "A cat is sleeping on the mat." H: "The cat is awake.")
    -   **Neutral:** The hypothesis is neither entailed nor contradicted by the premise. (e.g., P: "A cat is sleeping on the mat." H: "The cat is black.")
-   **Methods/Approaches:**
    -   **Traditional:** Feature engineering (lexical overlap, negation, syntactic features) with classifiers.
    -   **Deep Learning:** Sentence encoding models (using LSTMs, CNNs, or Transformers like BERT, RoBERTa) that learn to represent the premise and hypothesis and then compare these representations to make a classification. Often, the two sentences are fed into the model simultaneously (e.g., BERT with `[CLS] premise [SEP] hypothesis [SEP]`).
-   **Evaluation Metrics:**
    -   Accuracy.
-   **Key Challenges:**
    -   Requiring deep understanding of semantics, pragmatics, and common-sense reasoning.
    -   Handling negation, quantifiers, and complex logical structures.
    -   Lexical ambiguity and paraphrase recognition.
-   **Real-world Applications:**
    -   Evaluating the quality of text generation or machine translation systems.
    -   Information validation and fact-checking.
    -   Improving reading comprehension models.
    -   Underpinning more complex reasoning tasks.

---

## 13. Dialogue Systems / Chatbots

-   **Definition:** Systems designed to converse with humans using natural language.
-   **Input:** User utterance (text or speech).
-   **Output:** System response (text or speech).
-   **Types:**
    -   **Task-Oriented Dialogue Systems:** Aim to help users achieve a specific goal (e.g., booking a flight, making a reservation, technical support).
    -   **Open-Domain Chatbots (Chit-chat):** Aim to engage in general conversation, provide entertainment, or act as a companion, without a specific task to complete.
-   **Core Components (often for task-oriented systems):**
    1.  **Natural Language Understanding (NLU):**
        -   Interprets the user's utterance.
        -   Key sub-tasks: Intent recognition, slot filling (extracting specific parameters like date, location from the user's query).
    2.  **Dialogue State Tracker (DST) / Dialogue Management (DM):**
        -   Maintains the current state of the conversation (accumulated information, user goals).
        -   Decides the system's next action based on the dialogue state and policy (e.g., ask for more information, query a database, provide an answer).
    3.  **Natural Language Generation (NLG):**
        -   Converts the system's chosen action/response into a natural language utterance for the user.
-   **Methods/Approaches:**
    -   **Traditional:** Rule-based, finite-state machines for dialogue flow.
    -   **Deep Learning / Hybrid:**
        -   **NLU:** Often uses text classification (for intent) and NER/sequence labeling (for slot filling) with Transformer-based models.
        -   **DST/DM:** Can be rule-based, statistical, or increasingly using reinforcement learning or supervised learning on dialogue data.
        -   **NLG:** Template-based, statistical language models, or increasingly sophisticated generative models (e.g., GPT, T5) fine-tuned for response generation.
        -   **End-to-End Models:** Aim to learn the entire dialogue process from user input to system response directly from data, often using large Transformer-based Seq2Seq models.
-   **Evaluation Metrics:**
    -   **Task-Oriented:** Task completion rate, slot filling F1-score, dialogue length, user satisfaction (often via surveys).
    -   **Open-Domain:** Perplexity, BLEU/ROUGE (if reference responses exist), coherence, engagingness, fluency (often via human evaluation). Usability metrics like turn-level coherence, informativeness.
-   **Key Challenges:**
    -   Maintaining context over long conversations.
    -   Handling ambiguity and user corrections.
    -   Graceful error recovery.
    -   Generating diverse, engaging, and empathetic responses (for open-domain).
    -   Ensuring factual correctness and avoiding harmful or biased responses.
    -   Integrating external knowledge and APIs.
    -   Scalability and personalization.
-   **Real-world Applications:**
    -   Customer service automation.
    -   Virtual assistants (Siri, Alexa, Google Assistant).
    -   Information providing bots (e.g., weather, news).
    -   Educational companions.
    -   Mental health support (e.g., Woebot).

---
```

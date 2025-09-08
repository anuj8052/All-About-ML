# Module 7: Prompt Tuning and P-Tuning

## 1. The Paradigm Shift: Tune the Input, Not the Model

The PEFT methods we've discussed so far (LoRA, Adapters) still involve changing the model itself, either by adding new layers or by modifying the weight update process. Prompt-based tuning methods take an even more radical approach:

> What if we freeze the **entire** model and instead learn the best possible input prompt to guide the model to perform a specific task?

This is the core idea behind Prompt Tuning and its variants. Instead of manually engineering the perfect text prompt (a "hard prompt"), we let the model learn the perfect prompt in the continuous vector space of embeddings (a "soft prompt").

## 2. What are "Soft Prompts"?

-   **Hard Prompts:** These are the regular text prompts we write, e.g., "Summarize the following text: ...". The model's tokenizer converts these words into a fixed sequence of embeddings.
-   **Soft Prompts:** A soft prompt is a sequence of continuous vectors (embeddings) that are directly optimized through backpropagation. These vectors are not tied to any specific words in the model's vocabulary. They are **learnable parameters** that are prepended to the sequence of embeddings generated from the actual text input.

Think of it as finding the perfect "activation sequence" that puts the frozen LLM into the right "mode" to solve a specific task.

## 3. Prompt Tuning

Prompt Tuning is the most direct implementation of the soft prompt idea.

### How it Works:

1.  **Freeze the LLM:** The entire pre-trained language model is frozen. No weights are updated.
2.  **Initialize Soft Prompts:** A small sequence of, say, `k` random vectors (the soft prompt) is created. Each vector has the same dimension as the model's word embeddings. Let's call this trainable prompt matrix $P \in \mathbb{R}^{k \times d}$.
3.  **Prepend to Input:** For a given text input, it is first tokenized and converted to its embedding matrix, $E_{input} \in \mathbb{R}^{L \times d}$ (where L is the sequence length). The trainable prompt matrix $P$ is then concatenated at the beginning: $E_{combined} = [P; E_{input}] \in \mathbb{R}^{(k+L) \times d}$.
4.  **Train:** This combined embedding matrix is fed into the frozen LLM. The model generates an output, a loss is calculated, and backpropagation is used to update **only the parameters of the soft prompt matrix $P$**.

### Characteristics:
-   **Extreme Efficiency:** It is the most parameter-efficient method, often training as few as 0.001% of the model's parameters.
-   **Simplicity:** It's conceptually simple and easy to implement.
-   **Performance:** It performs very well on large models (10B+ parameters), often matching the performance of full fine-tuning. However, it can struggle with smaller models.

## 4. P-Tuning

P-Tuning (and its successor, P-Tuning v2) identified a weakness in the original Prompt Tuning: a soft prompt applied only at the input layer might not have enough influence on the model's deeper layers.

### How it's Different from Prompt Tuning:

P-Tuning also uses learnable prompt embeddings, but it treats them as "prompt tokens" that can be inserted at **multiple layers** of the Transformer, not just at the beginning. This allows the learned prompt to influence the model's internal computations at various stages, giving it more expressive power to steer the model's behavior.

While the original P-Tuning used a small LSTM to generate the prompt embeddings, the more modern P-Tuning v2 (which is what people usually refer to now) applies this "deep" prompting idea more directly, inserting trainable prompt vectors at each layer.

### Characteristics:
-   **More Powerful:** By influencing the model at every layer, it is more powerful and stable than basic Prompt Tuning, especially on smaller models and more complex tasks (like sequence tagging).
-   **Still Very Efficient:** While it adds more trainable parameters than basic Prompt Tuning, it is still exceptionally parameter-efficient compared to LoRA or full fine-tuning.

## 5. Pros and Cons of Prompt-based Methods

### Pros:
-   **Highest Parameter Efficiency:** They require the fewest trainable parameters of all PEFT methods.
-   **Tiny Checkpoints:** The saved "model" for a task is just the small soft prompt matrix, which can be only a few kilobytes in size. This makes storing and sharing task "models" incredibly easy.

### Cons:
-   **Interpretability:** The learned soft prompt vectors do not correspond to human-readable text, making them difficult to interpret.
-   **Stability:** Can sometimes be less stable to train compared to methods like LoRA.
-   **Performance Variance:** Performance can be sensitive to the choice of hyperparameters, like the length of the soft prompt. For many tasks, especially on smaller to medium-sized models, LoRA often provides a better balance of performance and efficiency.

# Module 6: Adapter-based Fine-Tuning

## 1. The Concept of Adapters

Before LoRA became the dominant PEFT technique, **Adapters** were one of a few pioneering methods that demonstrated the viability of parameter-efficient fine-tuning. The core concept is intuitive: instead of modifying the existing weights of a pre-trained model, we can **insert** new, small, and trainable neural network modules—called "adapters"—into the model's architecture.

Think of it like adding a small, specialized hardware card to a computer's motherboard. The motherboard (the pre-trained model) remains unchanged, but the new card (the adapter) adds new functionality. For each new task, you just train a new, lightweight adapter.

## 2. Adapter Architecture

Adapters are typically small, feed-forward neural networks that are injected into the layers of a Transformer. A common strategy is to add an adapter module after both the Multi-Head Attention sub-layer and the Feed-Forward Network (FFN) sub-layer within each Transformer block.

### The "Bottleneck" Structure

An adapter module itself usually has a "bottleneck" architecture to ensure it has very few parameters:

1.  **Down-Projection:** A linear layer that projects the high-dimensional input (e.g., the Transformer's hidden dimension, $d=768$) down to a much smaller dimension (the bottleneck dimension, $m=64$).
2.  **Non-linearity:** An activation function like ReLU or GeLU is applied.
3.  **Up-Projection:** A linear layer that projects the small dimension back up to the original hidden dimension ($m \to d$).
4.  **Residual Connection:** A residual or "skip" connection is added from the input of the adapter to its output. This ensures that if the adapter's weights are initialized to near-zero, the adapter initially has no effect on the model's output, which helps stabilize training.

### Diagrammatic Representation

Here's how an adapter fits inside a Transformer block:

```
      Input (from previous layer)
           |
           v
+------------------------+
| Multi-Head Attention   |
+------------------------+
|        |               |
| (Residual Connection)  |
|        v               |
|      Add & Norm        |------>+-----------------+
|        |               |      | Adapter Module  |
|        |               |      |  (Trainable)    |
|        |               |      +-----------------+
|        v               |               |
| +--------------------+ | <-------------+ (Output added back)
| | Feed-Forward Network | |
| +--------------------+ |
|        |               |
| (Residual Connection)  |
|        v               |
|      Add & Norm        |------>+-----------------+
|        |                      | Adapter Module  |
|        |                      |  (Trainable)    |
|        v                      +-----------------+
|                                        |
+------> Output (to next layer) <--------+ (Output added back)
```

## 3. How Adapters Work

The process is straightforward:

1.  **Freeze the Base Model:** All the original weights of the pre-trained LLM are frozen and are not updated during training.
2.  **Train Only Adapters:** Only the weights of the newly inserted adapter modules are trainable.
3.  **Modified Forward Pass:** The input, $h$, that comes out of a standard transformer sub-layer (like attention or FFN) is passed through the adapter module in parallel via a skip connection. The output of the adapter is then added back to the original output.
    $$ h_{new} = h + f_{adapter}(h) $$
    where $f_{adapter}$ is the function computed by the adapter module.

## 4. Pros and Cons of Adapters

### Pros:
-   **High Parameter Efficiency:** Like other PEFT methods, adapters drastically reduce the number of trainable parameters (e.g., to ~1-4% of the original model size).
-   **Clean Separation of Concerns:** The knowledge for a new task is cleanly encapsulated within the adapter modules. This makes it conceptually easy to manage and deploy different task-specific models.
-   **Modularity:** You can train adapters for many tasks and "plug them in" as needed without having to modify the base model.

### Cons:
-   **Inference Latency:** This is the most significant drawback of the adapter approach. Because you are adding new layers to the model, you are increasing the number of sequential computations that must be performed during inference. This can make the model slower compared to the original model or methods like LoRA (which can be merged back into the base weights to avoid any latency).
-   **Architectural Complexity:** Deciding exactly where to place the adapters and what their internal architecture should be can be complex and may require some experimentation.
-   **Superseded by LoRA:** For many use cases, LoRA has become more popular as it often achieves similar or better performance without introducing any inference latency, making it a more practical choice. However, adapters remain an important and foundational PEFT technique.

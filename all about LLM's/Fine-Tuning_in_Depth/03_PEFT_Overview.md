# Module 3: Overview of Parameter-Efficient Fine-Tuning (PEFT)

## 1. What is PEFT?

**Parameter-Efficient Fine-Tuning (PEFT)** represents a collection of techniques designed to adapt large pre-trained models to new tasks without having to train all of the model's parameters. The core idea is simple yet powerful:

> Freeze the vast majority of the pre-trained model's weights (which can be billions of parameters) and train only a very small number of new or existing parameters (typically <1% of the total).

This approach dramatically lowers the barrier to entry for fine-tuning, making it accessible to users with limited computational resources.

## 2. Why is PEFT a Game-Changer?

PEFT directly addresses the significant drawbacks of Full Fine-Tuning (FFT):

-   **Reduced Computational Cost:** By training only a fraction of the parameters, PEFT significantly reduces the memory (VRAM) required for gradients and optimizer states. This makes it possible to fine-tune massive models (7B, 13B, or even larger) on a single consumer or prosumer GPU.
-   **Faster Training:** Fewer parameters to update means each training step is faster.
-   **Efficient Storage:** Since the original pre-trained model is frozen and shared, you only need to store the small number of trained PEFT parameters for each new task. This is a difference of storing a few megabytes (for PEFT) versus many gigabytes (for FFT) for each task.
-   **Mitigation of Catastrophic Forgetting:** The original knowledge of the pre-trained model is preserved because its weights are not being updated. The model learns the new task through the small set of trainable parameters, effectively acting as a small "patch" or "plugin" on top of its existing knowledge.
-   **Better Performance on Small Datasets:** By constraining the number of trainable parameters, PEFT acts as a form of regularization, which can prevent overfitting when fine-tuning on smaller datasets.

## 3. A High-Level Summary of Key PEFT Techniques

PEFT methods can be broadly categorized based on how they achieve parameter efficiency. We will explore these in detail in the upcoming modules.

### a) Additive / Reparameterization Methods

These methods add new, small, trainable components to the frozen base model. The most popular and effective of these is a reparameterization technique:

-   **LoRA (Low-Rank Adaptation):** Instead of updating the original weight matrix, LoRA trains two much smaller, low-rank matrices whose product approximates the full weight update. This is the current state-of-the-art for PEFT in many scenarios.
-   **QLoRA (Quantized LoRA):** A highly optimized version of LoRA that further reduces memory by quantizing the base model's weights to 4-bit precision, making it possible to fine-tune huge models on a single GPU.

### b) Adapter-based Methods

This was one of the earliest and most intuitive PEFT approaches:

-   **Adapters:** Small, fully-connected neural network "bottleneck" layers are inserted between the existing layers of the transformer architecture. During fine-tuning, only the weights of these new adapter layers are trained.

### c) Prompt-based Methods

These methods don't touch the model's weights at all. Instead, they focus on manipulating the input to the model:

-   **Prompt Tuning:** A sequence of special "soft prompt" vectors (or "virtual tokens") is prepended to the input embedding sequence. The model learns the optimal values for these prompt vectors through backpropagation, effectively steering the frozen model's behavior towards the desired task.
-   **P-Tuning:** An improvement over prompt tuning that also uses trainable prompt embeddings but inserts them at various layers of the model for greater stability and performance.

These techniques offer a spectrum of trade-offs between performance, efficiency, and complexity. In the next modules, we will dissect each of these to understand their inner workings.

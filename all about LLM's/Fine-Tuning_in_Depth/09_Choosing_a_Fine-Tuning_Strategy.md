# Module 9: Choosing a Fine-Tuning Strategy

## 1. Introduction

We've explored a variety of fine-tuning techniques, from the resource-intensive Full Fine-Tuning (FFT) to the highly efficient QLoRA. The "best" method is not universal; it depends entirely on your specific project goals and constraints. This guide provides a framework for making an informed decision.

## 2. Key Decision Factors

Before choosing a method, you must assess your situation based on these key factors:

1.  **GPU Resources (VRAM):** This is often the most significant constraint. How much GPU memory do you have access to?
2.  **Task Performance:** Is it critical to squeeze out every last drop of performance, or is "very good" performance acceptable?
3.  **Number of Tasks:** Are you adapting a model for a single, primary task, or do you need to create dozens of different task-specific models?
4.  **Inference Latency:** Is the speed of the final, deployed model a critical concern?
5.  **Dataset Size:** How large and high-quality is your task-specific dataset?

## 3. A Decision Flowchart for Choosing a Method

Here is a step-by-step thought process to guide your choice.

**Question 1: What is your available GPU VRAM?**

-   **I have a large, multi-GPU cluster (e.g., >80GB VRAM per GPU).**
    -   You *can* use **Full Fine-Tuning**. Consider it only if you have a very large, high-quality dataset and initial tests with PEFT methods don't meet your absolute performance requirements. Even here, **QLoRA** on a larger base model often yields better results than FFT on a smaller one.
    -   **Recommendation:** Start with QLoRA. Only consider FFT if absolutely necessary.

-   **I have a high-end single GPU (e.g., 24GB - 48GB VRAM).**
    -   **QLoRA** is your best choice for fine-tuning large models (13B to 70B, depending on the exact GPU). It provides the best balance of performance and memory usage in this tier.
    -   **LoRA** (without quantization) is a great option for smaller models (e.g., <= 7B) where the full 16-bit base model can fit in memory alongside the training overhead.
    -   **Recommendation:** Default to **QLoRA**.

-   **I have a consumer-grade GPU (e.g., 8GB - 16GB VRAM).**
    -   **QLoRA** is essentially your only option for fine-tuning modern, powerful language models (e.g., 7B parameter models). It is specifically designed for this scenario.
    -   **Recommendation:** Use **QLoRA**.

**Question 2: Based on your chosen method, how do other factors influence tuning?**

-   **If you need to support many tasks...**
    -   Any **PEFT** method (QLoRA, LoRA, Adapters) is vastly superior to FFT. The ability to store task-specific adaptations as small, megabyte-sized files is a massive advantage over storing full multi-gigabyte models for each task.

-   **If inference latency is critical...**
    -   **LoRA and QLoRA are superior.** Their learned weights can be merged back into the base model's weights after training. This means the final, deployed model has the exact same architecture and speed as the original base model—zero latency is introduced.
    -   **Adapters** will introduce latency because they add extra layers that must be processed during every forward pass.

-   **If you have a very small dataset...**
    -   PEFT methods, especially those with very few trainable parameters like **LoRA with a small rank (r)** or **Prompt Tuning**, can be more robust against overfitting compared to FFT.

## 4. Summary Table and General Recommendations

| Method                  | Key Characteristic                                         | When to Use It                                                                                              |
| ----------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Full Fine-Tuning**    | Trains all 100% of parameters.                             | **Rarely.** Only if you have massive GPU resources and PEFT methods have proven insufficient for performance. |
| **QLoRA**               | Trains LoRA adapters on a 4-bit quantized base model.      | **The default choice for most users.** Maximizes memory efficiency, enabling large models on single GPUs.    |
| **LoRA**                | Trains low-rank adapters on a 16-bit base model.           | When you have enough VRAM to hold the 16-bit model and want to avoid potential (minor) quantization errors. |
| **Adapters**            | Inserts small, trainable layers into the model.            | When inference latency is not a concern and you prefer the modularity of separate adapter layers.            |
| **Prompt Tuning**       | Freezes the entire model, only trains input "soft prompts". | When you have a very large model (10B+) and a relatively simple task. Good for extreme parameter efficiency.   |

**Final Recommendation:** For 95% of users in 2023-2024, the best strategy is to **pick the largest, best-performing base model you can fit on your GPU using QLoRA, and fine-tune it.** This approach consistently delivers the best performance for a given hardware constraint.

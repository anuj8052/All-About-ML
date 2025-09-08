# Module 10: Practical Example with Hugging Face

## 1. Introduction

This module provides a hands-on, step-by-step tutorial for fine-tuning a pre-trained language model using **QLoRA**. We will use the powerful libraries from the Hugging Face ecosystem: `transformers` for models, `datasets` for data handling, and `peft` for our parameter-efficient fine-tuning technique.

Our goal is to fine-tune a small model on a simple instruction-following dataset. This code can be run in a Google Colab notebook with a T4 GPU.

## 2. Prerequisites

First, you need to install the necessary libraries.

```bash
pip install -q transformers datasets accelerate peft bitsandbytes trl
```
-   `transformers`: The core library for models and tokenizers.
-   `datasets`: For loading and processing data.
-   `accelerate`: Simplifies running PyTorch code on any infrastructure (CPU, GPU, multi-GPU).
-   `peft`: The Parameter-Efficient Fine-Tuning library, containing our LoRA implementation.
-   `bitsandbytes`: Required for quantization (loading the model in 4-bit).
-   `trl`: The Transformer Reinforcement Learning library, which provides a convenient `SFTTrainer`.

## 3. Step-by-Step Fine-Tuning

### Step 1: Load the Dataset

We'll use the `mlabonne/guanaco-llama2-1k` dataset, which is a small, clean dataset of 1,000 instructions and responses, perfect for a quick demonstration.

```python
from datasets import load_dataset

# Load the dataset
dataset_name = "mlabonne/guanaco-llama2-1k"
dataset = load_dataset(dataset_name, split="train")

# You can inspect the data
print(dataset[0])
# Expected output: {'text': '<s>[INST] Explain the importance of low-rank adaptation in fine-tuning large language models. [/INST] Low-rank adaptation...'}
```

### Step 2: Load the Model and Tokenizer

We will use `NousResearch/Llama-2-7b-chat-hf` as our base model. To make it fit in memory, we will load it in 4-bit precision using `bitsandbytes`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model and tokenizer names
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Configuration for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Use NF4 quantization
    bnb_4bit_compute_dtype=torch.float16, # Use float16 for computations
    bnb_4bit_use_double_quant=True, # Use double quantization
)

# Load the model with the specified quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # Automatically map the model to the available devices
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # Set padding token to end-of-sequence token
```

### Step 3: Configure LoRA (PEFT)

Now, we define our LoRA configuration using the `peft` library. We'll target the query and value projection matrices in the attention layers, as is common practice.

```python
from peft import LoraConfig

# LoRA configuration
lora_config = LoraConfig(
    r=16, # Rank of the update matrices.
    lora_alpha=32, # Alpha parameter for scaling.
    target_modules=["q_proj", "v_proj"], # Modules to apply LoRA to.
    lora_dropout=0.05, # Dropout probability.
    bias="none", # Bias training.
    task_type="CAUSAL_LM" # Task type.
)
```

### Step 4: Set Up the Trainer

We'll use the `SFTTrainer` from the `trl` library, which is designed for supervised fine-tuning on instruction-like datasets.

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# Training arguments
training_args = TrainingArguments(
    output_dir="./llama2-7b-qlora-finetuned", # Directory to save the model
    per_device_train_batch_size=4, # Batch size
    gradient_accumulation_steps=4, # Gradient accumulation
    learning_rate=2e-4, # Learning rate
    num_train_epochs=1, # Number of training epochs
    logging_steps=10, # Log every 10 steps
    fp16=True, # Use 16-bit precision
    max_grad_norm=0.3, # Max gradient norm
    max_steps=-1, # Number of training steps (set to -1 for epochs)
    warmup_ratio=0.03, # Warmup ratio
    group_by_length=True, # Group sequences of similar length
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text", # The field in the dataset containing the text
    max_seq_length=512, # Max sequence length
    tokenizer=tokenizer,
    args=training_args,
)
```

### Step 5: Train the Model

This is the simplest step! Just call the `train()` method.

```python
# Start training
trainer.train()
```

### Step 6: Save the model

```python
# Save the fine-tuned model
trainer.save_model("./llama2-7b-qlora-finetuned-model")
```

### Step 7: Inference

Let's test our fine-tuned model with a prompt.

```python
from peft import PeftModel
import torch

# Reload the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Merge the base model with the LoRA adapter
model = PeftModel.from_pretrained(base_model, "./llama2-7b-qlora-finetuned-model")
model = model.merge_and_unload()

# Reload tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare the prompt
prompt = "What is the capital of France?"
inputs = tokenizer(f"<s>[INST] {prompt} [/INST]", return_tensors="pt", return_attention_mask=False)

# Generate a response
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

This complete example shows how accessible fine-tuning has become. With just a few lines of code, you can adapt a powerful LLM to your specific needs on a single GPU.

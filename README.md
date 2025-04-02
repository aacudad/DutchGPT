# 🇳🇱 Dutch-Language Fine-Tuned LLMs: Gemma3 4b & LLaMA3.2 3b

<!-- ![License](https://img.shields.io/badge/license-MIT-blue.svg) -->
![Models](https://img.shields.io/badge/models-fine--tuned-orange)
<!-- ![Language](https://img.shields.io/badge/language-Dutch-blue) -->
![Status](https://img.shields.io/badge/status-Experimental-yellow)

> 🧠 Fine-tuning Gemma3 4b and LLaMA3.2 3b to **speak better Dutch** and handle tasks like **summarization** and **understanding** in a low-resource setting.

---

## 📌 Table of Contents
- [🔍 Introduction](#-introduction)
- [🎯 Motivation](#-motivation)
- [⚙️ Environment Setup](#️-environment-setup)
- [🚀 Inference Guide](#-inference-guide)
- [🧪 Fine-Tuning Approach](#-fine-tuning-approach)
- [📂 Data Collection](#-data-collection)
- [🧾 Code Structure](#-code-structure)
- [📊 Results (Template)](#-results-template)
- [🔍 Observations (Template)](#-observations-template)
- [✅ Conclusion](#-conclusion)
- [📚 References](#-references)
- [📬 Contact](#-contact)

---

## 🔍 Introduction

This repository contains **two Dutch-enhanced LLMs**:

- 🟢 **Gemma3 4b** – already decent in Dutch, further optimized.
- 🔵 **LLaMA3.2 3b** – initially poor at Dutch, now upgraded.

Both models were fine-tuned with [Unsloth](https://unsloth.ai) and are hosted on:
- 🐑 [Hugging Face](https://huggingface.co/aacudad)
- 🦙 [Ollama](https://ollama.com/aacudad)


## 🎯 Motivation

Why this repo exists:

- 🟨 Improve Dutch performance of **LLaMA3.2 3b**, which underperforms out of the box.
- 🟩 Push **Gemma3 4b** to a higher level of Dutch fluency.
- 🧪 Use efficient methods (LoRA, QLoRA, Flash Attention) for resource-aware fine-tuning.
- 🎓 Make it approachable for **non-technical users** via simple deployment and usage instructions.

---

## ⚙️ Environment Setup

### 🧰 Prerequisites

- **OS**: Linux, macOS, or Windows  
- **Python**: 3.8+  
- **Hardware**: 8GB RAM minimum (GPU strongly recommended for local inference)

---

### 🐍 Install Python Dependencies (⚠️ Only if you want to explore or modify the code)

> 🧑‍💻 This section is **only relevant if you want to dive into the codebase**, perform custom training, or run the models using Python (e.g., via scripts like `GEMMA.py` or `llama3b.py`). 

> ❌ **You do NOT need this** if you only want to run inference via **Ollama** or **Hugging Face**.




#### ✅ Recommended: Use `uv` for reproducible environments

[`uv`](https://github.com/astral-sh/uv) is a fast and modern Python package manager that simplifies dependency management and improves reproducibility. Here’s how to use it:

1. **Install `uv`**

   Visit [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/) and follow the installation instructions for your OS.

2. **Clone the repository**  
   ```bash
   git clone https://github.com/aacudad/DutchGPT
   cd DutchGPT
   ```

3. **Add the virtual environment**  
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install all dependencies from `uv.lock`**  
   ```bash
   uv sync
   ```

> ⚠️ `uv` is the recommended method for reproducibility, performance, and consistency across environments.

#### 🐢 Alternatively: Use `pip` (less preferred)

If you prefer the classic `pip` approach:

```bash
git clone https://github.com/aacudad/DutchGPT
cd DutchGPT
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

> ☑️ Your `requirements.txt` should include:
```
torch
transformers
accelerate
unsloth
```

---

### 📥 Downloading Models

- **Gemma3 4b**  
  - 🦙 Ollama: [https://ollama.com/aacudad/gemma-3-DUTCH](https://ollama.com/aacudad/gemma-3-DUTCH)  
  - 🤗 Hugging Face: [https://huggingface.co/aacudad/gemma-3-finetune](https://huggingface.co/aacudad/gemma-3-finetune)

- **LLaMA3.2 3b**  
  - 🦙 Ollama: [LINK_PLACEHOLDER](#)  
  - 🤗 Hugging Face: [LINK_PLACEHOLDER](#)

---

## 🚀 Inference Guide

### 🦙 Using Ollama

### 🔧 Installing Ollama

1. **Download Ollama**:  
   👉 [ollama.ai/download](https://ollama.ai/download)

2. **Install it** and follow instructions to get started.

3. **Pull and run the model**:

```bash
ollama pull <model-name>
ollama run <model-name>
```

Once the model starts, you’ll see the following prompt and can begin entering your messages:

```bash
>>> Send a message (/? for help)
```

📌 Example:

```bash
ollama run aacudad/gemma-3-DUTCH 
>>> Send a message (/? for help)
```

---

### 🤗 Using Hugging Face (Python)

????NEEDS CHECKING???????

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("your-hf-username/gemma3-4b")
model = AutoModelForCausalLM.from_pretrained("your-hf-username/gemma3-4b")

prompt = "Geef een samenvatting van de onderstaande tekst in het Nederlands: ..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🧪 Fine-Tuning Approach

All models were fine-tuned with [Unsloth](https://unsloth.ai) using:

- **🔁 LoRA (Low-Rank Adaptation)**  
  Efficiently fine-tunes a few layers without updating the full model.  
  📄 [Paper](https://arxiv.org/abs/2106.09685)

- **⚡ QLoRA**  
  Combines quantization with LoRA for low-resource hardware.  
  📄 [Paper](https://arxiv.org/pdf/2305.14314)

- **⚡ Flash Attention**  
  Fast and memory-efficient attention module for large models.  
  📄 [Repo](https://github.com/HazyResearch/flash-attention)

---

## 📂 Data Collection

*To be added.*

This section will describe:
- 📚 Data sources
- 🧹 Preprocessing steps
- 📊 Dataset size and distribution

---

## 🧾 Code Structure

This repository includes two Python scripts that handle model loading, quantized fine-tuning (via LoRA/QLoRA), and inference using the [Unsloth](https://unsloth.ai) framework. Both scripts are optimized for memory efficiency and Dutch-language task performance.

| File         | Description                                         |
|--------------|-----------------------------------------------------|
| `GEMMA.py`   | Fine-tuning & inference pipeline for **Gemma3 4b**  |
| `llama3b.py` | Fine-tuning & inference pipeline for **LLaMA3.2 3b**|



### 🟢 [Gemma3-4b](https://github.com/your-username/your-repo/blob/main/docs/example.txt) – Gemma3 4b (Dutch)

This script loads and prepares the `unsloth/gemma-3-4b-it` model using Unsloth's `FastModel`.

#### 🧩 Features:
- Loads **Gemma3 4b** with **4-bit quantization** (`bnb-4bit`) to reduce VRAM usage.
- Enables **LoRA adapters** for efficient parameter-efficient fine-tuning.
- Supports **4096-token context window**.
- Built to run on consumer GPUs (e.g., A6000).

#### 🔍 Example Code Logic:

```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it",
    max_seq_length = 4096,
    load_in_4bit = True,
    full_finetuning = False,
)
```

- `load_in_4bit=True`: Enables **quantized inference/fine-tuning** with 4x memory savings.
- `full_finetuning=False`: Only LoRA adapters are trained.

#### 🛠️ Use Case:
Efficient fine-tuning or inference with Gemma3 4b in **Dutch**, with **low memory** footprint and fast startup time.



### 🔵 [LLaMa3.2-3b](https://github.com/your-username/your-repo/blob/main/docs/example.txt) – LLaMA3.2 3b (Dutch)

This script loads the `unsloth/Llama-3.2-3B` model using `FastLanguageModel`, which supports larger context sizes and broader model support.

#### 🧩 Features:
- Loads **LLaMA3.2 3b** with 4-bit quantization.
- Supports **very long context length (16,384 tokens)**.
- Uses Unsloth’s automatic **RoPE scaling** for long documents.
- Compatible with both inference and LoRA-based fine-tuning.

#### 🔍 Example Code Logic:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B",
    max_seq_length = 16384,
    load_in_4bit = True,
    dtype = None,
)
```

- `max_seq_length=16384`: Enables summarizing or processing long Dutch documents.
- `load_in_4bit=True`: Optimized for memory-efficient execution on a single GPU.
- `dtype=None`: Auto-selects the best floating-point type (e.g., `bfloat16`, `float16`).

#### 🛠️ Use Case:
High-throughput inference or long-form fine-tuning with LLaMA3.2 3b using **long context sequences**, useful for summarizing articles, legal texts, etc.

---

### ⚙️ Shared Best Practices

- ✅ Both scripts use **Unsloth's quantized, efficient backends**.
- 🔁 LoRA adapters allow rapid fine-tuning with minimal compute.
- 🧠 All models were trained on an **NVIDIA RTX A6000 (48 GB VRAM)**, enabling large-batch, long-context training.
- 🧪 Fully reproducible and modifiable—designed for research and production.

---

## 📊 Results (Template)

> Results and graphs will be filled in post-evaluation.

### 📈 Example Layout

- **Model**: Gemma3 4b / LLaMA3.2 3b  
- **Task**: Summarization in Dutch  
- **Metrics**: ROUGE, BLEU, latency  
- **Graph**:  
  ![Benchmark](path/to/benchmark_graph.png)

#### 📌 Analysis
- 🔸 Observation 1  
- 🔸 Observation 2  
- 🔸 Observation 3  

---

## 🔍 Observations (Template)

Fill this section after experimentation:

- ⚠️ Weaknesses observed  
- ✅ Strengths gained post-fine-tuning  
- 🧠 Qualitative examples

---

## ✅ Conclusion

This project shows how to fine-tune, deploy, and evaluate LLMs for low-resource languages like Dutch. With accessible tools like Ollama and Hugging Face, and efficient methods like LoRA/QLoRA, anyone can:

- Adapt LLMs to a specific domain/language.
- Run models efficiently on consumer hardware.
- Share and benchmark results transparently.

---

## 📚 References

- [LoRA](https://arxiv.org/abs/2106.09685)  
- [QLoRA (Hugging Face)](https://huggingface.co/blog/quantized-lora)  
- [Flash Attention](https://github.com/HazyResearch/flash-attention)  
- [Unsloth](https://unsloth.ai)

---

## 📬 Contact

Feel free to reach out with questions or collaboration ideas:  
**Adnane Acudad** – aacudad@tudelft.nl
**Akram Chakrouni** – achakrouni@tudelft.nl

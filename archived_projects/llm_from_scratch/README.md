# LLM Experiment: Building a Language Model from Scratch (Archived)

## Overview

This subproject represents an experimental attempt to build a basic large language model (LLM) architecture from scratch using PyTorch. The goal was to gain hands-on experience with the internal mechanics of transformer-based models by manually implementing core components.

## Project Motivation

Large language models (LLMs) like GPT and BERT have significantly advanced the field of natural language processing. Rather than treating these models as black boxes, this project was initiated to develop a foundational understanding of their architecture, including:

- Embedding layers
- Self-attention mechanisms
- Layer normalization
- Model initialization and parameter tuning

## Current Status

**⚠️ Project Status: Incomplete and Archived**  
This project was paused early in development and is no longer actively maintained. The current work consists of architectural prototyping, dimensionality checks, and some scaffolding for transformer components.

## Reason for Archival

Following clarification from the course teaching assistant, it became evident that fine-tuning existing pretrained models was both permissible and far more efficient for our project's objectives. Given the high computational and temporal costs of training a language model from scratch, we shifted focus toward fine-tuning and benchmarking existing models in Dutch—delivering more meaningful and actionable results within the project timeline.

Despite this shift, the exploratory work done here represents a serious attempt at LLM development and is retained for its instructional and referential value.

## Context & Data

- `train_data.ipynb`:  
  This notebook demonstrates how the input data was ingested and formatted for early-stage language modeling experiments.
  
  **Key Steps:**
  - A raw Dutch text corpus was manually provided.
  - All line breaks were stripped to form a single continuous character stream.
  - A character-level vocabulary was generated dynamically from the text.
  - Text was encoded as integer token sequences using the vocabulary mapping (`char_to_idx` and `idx_to_char` dictionaries).
  - Basic train/validation split was performed by character index.

  This character-level encoding approach aligns with the goal of building a minimal LLM prototype from the ground up, where even tokenization is kept deliberately simple.

## Contents

- `model_init.ipynb`:  
  A notebook detailing:
  - Initial embedding and positional layers  
  - Scaled dot-product attention  
  - Parameter initialization tests  
  - Dimensionality validation and architectural sketches  

- `train_data.ipynb`:  
  Preprocessing logic for building a character-level dataset and vocabulary.

## Potential Future Work

Although currently archived, this prototype could be extended in the future to include:
- Full transformer block implementation
- Causal masking and autoregressive training
- Tokenization and dataset integration (e.g., WikiText)
- Full training loop with optimization and loss tracking
- Language modeling benchmark evaluation

## How to Use This

This is an exploratory, standalone notebook-based experiment and is not integrated with the main project pipeline. It serves as a reference or potential template for future LLM prototyping.

## License

This work is provided under the MIT License. See the root repository for license details.

## Attribution

This experiment was authored as part of a broader NLP research project. While inactive, it is included in this repository to document the scope of exploration and learning involved. Use and contributions are welcome under research or educational contexts.

---

*This project is part of a larger repository focused on fine-tuning and benchmarking multilingual NLP models. For results, benchmarks, and reproducible pipelines, refer to the main project README.*

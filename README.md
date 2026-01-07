# Next Word Prediction: LSTM vs Fine-Tuned LLM

## Overview
This project implements an end-to-end Next Word Prediction system using two different approaches:
1. A traditional LSTM-based language model
2. A transformer-based Large Language Model (LLM) fine-tuned from GPT-2

The objective of this project is to compare classical sequence models with modern transformer-based models in terms of contextual understanding, scalability, and deployment readiness.

---

## Project Highlights
- Implemented an LSTM-based next-word prediction model trained from scratch
- Fine-tuned a pretrained DistilGPT-2 model using causal language modeling
- Built a clean inference pipeline for both models
- Exposed the LLM via a FastAPI REST API
- Containerized the inference service using Docker
- Followed production-oriented project structure and best practices

---

## Models Implemented

### LSTM-Based Language Model
- Architecture: Embedding → LSTM → Dense (Softmax)
- Training: From scratch
- Dataset: Small domain-specific text corpus
- Purpose: Baseline model for comparison

Characteristics:
- Captures short-range dependencies
- Strongly dependent on dataset size and domain
- Limited long-context understanding

---

### Fine-Tuned LLM (DistilGPT-2)
- Architecture: Transformer with self-attention
- Training: Fine-tuning on pretrained weights
- Dataset: WikiText (general-domain corpus)
- Objective: Causal language modeling

Characteristics:
- Strong contextual awareness
- Handles long-range dependencies
- Generates fluent and coherent text
- Suitable for real-world deployment

---

## Model Comparison

| Feature | LSTM Model | Fine-Tuned LLM |
|------|-----------|---------------|
| Model Type | Recurrent Neural Network | Transformer |
| Training Method | From scratch | Fine-tuning |
| Pretraining | No | Yes |
| Context Length | Short | Long |
| Dependency Modeling | Sequential | Self-attention |
| Text Fluency | Moderate | High |
| Dataset Sensitivity | High | Low |
| Scalability | Limited | High |
| Deployment Readiness | Low | Production-ready |

---

## Example Output Comparison

Input Prompt:   "India is"

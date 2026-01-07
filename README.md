# Next Word Prediction System: LSTM vs Fine-Tuned LLM

## Overview
This project implements an end-to-end **Next Word Prediction system** using two approaches:

1. A traditional **LSTM-based language model** (baseline)
2. A **Transformer-based Large Language Model (DistilGPT-2)** fine-tuned using causal language modeling

The primary objective is to **compare classical sequence models with modern transformer-based models** and demonstrate a complete ML workflow from training to deployment.

---

## Complete Workflow

### Step 1: Baseline Model (LSTM)
- Built an LSTM-based language model from scratch
- Used tokenization, padding, and sequence modeling
- Trained on a small, domain-specific dataset
- Implemented next-word prediction using softmax output

Purpose:
- Acts as a baseline
- Demonstrates limitations of RNN-based language models

---

### Step 2: Advanced Model (Fine-Tuned LLM)
- Loaded pretrained DistilGPT-2 using Hugging Face Transformers
- Fine-tuned the model using **causal language modeling**
- Dataset used: WikiText (general-domain corpus)
- Used PyTorch + Hugging Face Trainer for training

Key idea:
- Instead of training from scratch, the model adapts existing language knowledge

---

### Step 3: Inference Pipeline
- Created a clean inference script (`inference.py`)
- Loads tokenizer and model weights
- Generates next-word predictions using controlled sampling:
  - top-k
  - top-p
  - temperature

This separates **training logic** from **inference logic**.

---

### Step 4: FastAPI Backend
- Built a REST API using FastAPI
- Exposed an endpoint for next-word prediction
- Accepts input text and returns generated continuation
- Designed for real-time inference

Endpoint:


---

### Step 5: Dockerization
- Created a Dockerfile for the FastAPI inference service
- Installed dependencies via requirements.txt
- Copied inference and API code into the container
- Exposed port 8000

The application is container-ready and platform-independent.

---

### Step 6: CI/CD with GitHub Actions
- Used GitHub Actions to build the Docker image
- Automatically pushed the image to Docker Hub
- No local Docker installation required
- Ensures reproducible and automated builds

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

Input Prompt: "India is"
LSTM Output:   India is a popular food item
LLM Output:    India is a diverse country with a rich cultural heritage and a rapidly growing technological ecosystem.

## Folder Structure

project-root/
│
├── lstm_model/
│ ├── training.ipynb
│ ├── inference.py
│ └── tokenizer.pkl
│
├── llm_model/
│ ├── llm_next_word_finetuning.py
│ ├── inference.py
│ ├── requirements.txt
│ └── saved_model/
│ └── .gitignore
│
├── api/
│ └── app.py
│
├── .github/
│ └── workflows/
│ └── docker-build.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore


---

## Tech Stack
- Python
- TensorFlow (LSTM)
- PyTorch
- Hugging Face Transformers
- FastAPI
- Docker
- GitHub Actions (CI/CD)
- Google Colab (training environment)

---

## Deployment Notes
- Model weights are not committed to GitHub due to size constraints
- `.gitignore` is used to exclude large artifacts
- Docker image is built and pushed automatically using GitHub Actions
- The image can be run on any Docker-supported platform

---

## Key Learnings
- Transformer models outperform LSTMs for language modeling tasks
- Fine-tuning pretrained LLMs is more effective than training from scratch
- Separating training, inference, and deployment improves maintainability
- CI/CD pipelines simplify and automate ML deployment


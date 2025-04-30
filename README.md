# 🐉 High Valyrian MiniGPT — Build GPT Language Model From Scratch

---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![WandB](https://img.shields.io/badge/W%26B-Integrated-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)



## 📖 Project Description

This project is a **full from-scratch implementation** of a **GPT-like transformer model**, trained to generate text in the fictional **High Valyrian language** (from *Game of Thrones* universe).
It was created as a **learning project** to deeply understand:
- The inner working of **Large Language Models (LLMs)**
- Every important component: **tokenizer**, **dataloader**, **transformer**, **training tricks**, **inference techniques**
- How **GPT** models are really built under the hood — **no high-level libraries like HuggingFace were used**.

✅ **Goal:** Build everything by hand, and **learn by building**.

✅ **Target Audience:** Anyone curious about LLM internals, researchers, students, and engineers.

---

## 🛠️ Tools and Libraries Used

| Tool/Library | Why we used it |
|:---|:---|
| **Python** | For clean, flexible model prototyping |
| **PyTorch** | To manually implement Transformer architecture with full control over layers |
| **Weights & Biases (wandb)** | For tracking training metrics, loss curves, debugging models easily |
| **TorchScript** | To export the final trained model for optimized production/inference use |
| **Matplotlib** (optional) | For visualization (loss curves, attention maps in future extensions) |

---

## 🏛️ Model Architecture

### ➡️ Components Implemented:
- **Byte Pair Encoding (BPE)** tokenizer (custom)
- **Causal Self-Attention** layer
- **Multi-Head Attention** with masking
- **Transformer Block** with pre-layernorm and MLP
- **Positional Embeddings**
- **GPT Decoder Head** (Linear layer to predict next token)

---

### 🧐 Model Hyperparameters:

| Hyperparameter | Value |
|:---------------|:------|
| Embedding dimension | 128 |
| Number of heads | 4 |
| Number of layers (Transformer blocks) | 4 |
| Dropout | 0.2 |
| Vocabulary Size | ~500 tokens (custom BPE tokenizer) |
| Context Length (Block Size) | 64 tokens |

---

### 🔥 How This Differs From GPT-2:

| Feature | GPT-2 (Original) | High Valyrian MiniGPT (This project) |
|:---|:---|:---|
| Model size | 117M+ parameters | ~1.5M parameters |
| Tokenizer | Trained on WebText, massive | Trained on small High Valyrian corpus |
| Layers/Heads | 12 layers, 12 heads | 4 layers, 4 heads |
| Dropout | Yes | Yes |
| Weight initialization | Gaussian Normal | Same (manual) |
| LayerNorm | Post-Attention | **Pre-Attention** (better for small models) |
| Training tricks | AdamW, cosine decay | Adam, weight decay, early stopping |
| Scaling | Billion-token corpus | Small educational corpus |

✅ **Architecturally faithful to GPT-2**, but scaled **down** for fast training and clarity.

---

## 🌟 Key Features

- True **causal masked attention** (can't see future tokens)
- Custom **top-p (nucleus) sampling** during inference
- Full **training, evaluation, checkpointing, and wandb logging**
- Exportable via **TorchScript** for future deployment
- Lightweight and easy to train on a **single GPU**

---

## 📈 WandB Training Logs

You can see the full training progress here:  
🔗 [**Project WandB Dashboard**](https://wandb.ai/suriya137prakash-arizona-state-university/high-valyrian-gpt)

(includes loss curves, model samples during training, and metrics)

---

## 🚀 How to Run the Project

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare dataset:
   - Add your corpus file to `data/High Valyrian.txt`.

3. Train BPETokenizer:
   ```bash
   python tokenizer/train_tokenizer.py
   ```

4. Train model:
   ```bash
   python train.py
   ```

5. Generate text:
   ```bash
   python inference/generate.py --prompt "valar morghulis"
   ```

6. Export to TorchScript:
   ```bash
   python scripts/export_torchscript.py
   ```

---

## 📜 Lessons Learned

- Building LLMs is about **small careful engineering choices**: masking, layer ordering, normalization, sampling.
- **Training tricks** (dropout, optimizer tweaks) make a **huge difference**.
- **Inference matters** as much as training — generation quality is the true test.
- **Tooling** (like wandb) is essential to **debug and understand** model behavior.
- Scaling up (to bigger GPTs) is mostly about **data, compute, and model depth** — the core principles stay the same.

---

# ✨ Final Thoughts

This project demonstrates not just how to *use* LLMs, but how to **build** them — **layer by layer**, **loss by loss**, **token by token**.

> If you truly understand something, you can build it yourself.

That is the spirit of this project. 👨‍🔧🔥


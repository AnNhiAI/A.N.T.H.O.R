# A.N.T.H.O.R

**AnNhi Thrifty High-Optimization Research**

Training a Large Language Model from scratch on a single NVIDIA T4 GPU (16GB VRAM).

## 🎯 Mission

Democratize LLM training by proving that a capable language model can be trained from scratch on consumer-grade hardware. This project explores the frontier of **extreme efficiency** — combining cutting-edge optimization techniques, architectural innovations, and training strategies to maximize model quality under severe resource constraints.

## 🧠 Why A.N.T.H.O.R?

Most LLMs require hundreds of GPUs and millions of dollars to train. A.N.T.H.O.R challenges this paradigm by asking: **What can we achieve with just one T4 GPU?**

This is both a research playground and a practical experiment in:

- **Memory efficiency** — fitting large models into 16GB VRAM
- **Compute efficiency** — maximizing learning per FLOP
- **Data efficiency** — getting the most out of limited pretraining data
- **Architecture exploration** — testing novel designs that prioritize parameter efficiency

## 🔧 Technical Stack

| Component | Approach |
|-----------|----------|
| **Architecture** | Decoder-only Transformer with modern enhancements (RoPE, SwiGLU, RMSNorm, Grouped-Query Attention) |
| **Precision** | Mixed precision (FP16/BF16), gradient checkpointing, possibly quantization (FP8/INT8) |
| **Optimization** | Memory-efficient optimizers (AdamW with bitsandbytes 8-bit, GaLore, LoRA-style training) |
| **Parallelism** | Single-GPU with micro-batch accumulation, activation checkpointing |
| **Data Processing** | Custom tokenizer, efficient dataloading with memory-mapped datasets |
| **Monitoring** | Weights & Biases / TensorBoard for tracking runs |

## 🚀 Features & Techniques

### Memory Optimization
- [ ] Activation checkpointing (gradient checkpointing)
- [ ] CPU offloading for optimizer states
- [ ] 4-bit / 8-bit quantization (QLoRA-style)
- [ ] Parameter-efficient fine-tuning (LoRA, DoRA)
- [ ] GaLore (Gradient Low-Rank Projection)

### Training Efficiency
- [ ] Micro-batch accumulation with gradient scaling
- [ ] Dynamic batch sizing
- [ ] Learning rate scheduling (cosine decay with warmup)
- [ ] Gradient clipping and normalization
- [ ] Stochastic depth / layer dropout

### Data & Tokenization
- [ ] Train custom BPE / Unigram tokenizer
- [ ] Efficient dataset streaming
- [ ] Data deduplication and quality filtering
- [ ] Curriculum learning (easy → hard data ordering)

### Model Architecture
- [ ] Rotary Position Embeddings (RoPE)
- [ ] SwiGLU activation in FFN layers
- [ ] Root Mean Square Normalization (RMSNorm)
- [ ] Grouped-Query Attention (GQA)
- [ ] Optional: Mixture of Experts (MoE) layers

## 📊 Project Status

🚧 **Early Development** — The project is in initial research and prototyping phase.

- [ ] Literature review & technique selection
- [ ] Architecture design & scaling laws analysis
- [ ] Tokenizer training
- [ ] Data pipeline construction
- [ ] Small-scale training runs (proof of concept)
- [ ] Full training run
- [ ] Evaluation & benchmarking
- [ ] Release model weights

## 🔬 Research Log

Key findings and experiments will be documented as the project progresses.

## 📓 Notebooks (Google Colab)

The entire project is structured as **Jupyter Notebooks** (`.ipynb`) designed to run on **Google Colab** (free T4 GPU tier). Each notebook is self-contained with installation, setup, and execution cells — just open in Colab and run.

```
ANTHOR/
├── README.md
├── README.vi.md
├── 01_tokenizer.ipynb       # Train BPE/Unigram tokenizer
├── 02_data_pipeline.ipynb   # Data processing & streaming
├── 03_model.ipynb           # Model architecture implementation
├── 04_train.ipynb           # Training loop & optimization
├── 05_eval.ipynb            # Evaluation & benchmarks
├── utils.ipynb              # Shared utility functions
└── configs/                 # Training & model config files
```

## 🖥️ Hardware Requirements

- **Minimum:** NVIDIA T4 (16GB VRAM)
- **Target:** Single GPU setup
- **RAM:** 32GB+ system memory recommended
- **Storage:** 200GB+ free space for datasets & checkpoints

## 📚 References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [QLoRA: Efficient Finetuning of Quantized Language Models](https://arxiv.org/abs/2305.14314)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

## 📄 License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by AnNhi — because great things start with limited resources.*

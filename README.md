# Phased Continual Pre-Training (CPT) with LoRA

A production-ready framework for **phased fine-tuning** of large language models using LoRA (Low-Rank Adaptation) with DeepSpeed optimization. Designed for multi-GPU training on models like Kwaipilot/KAT-Dev (32B parameters).

## 🎯 Overview

This framework implements a **curriculum learning approach** where the model is trained sequentially across multiple phases, each building upon the previous:

| Phase | Name | Purpose | Default Epochs |
|-------|------|---------|----------------|
| 1 | **Foundation** | Core codebase understanding | 3 |
| 2 | **Evolution** | Code evolution patterns | 2 |
| 3 | **PR Mastery** | Pull request and review patterns | 2 |

### Key Features

- 🔄 **Phased Training**: Sequential fine-tuning across multiple curriculum stages
- ⚡ **DeepSpeed Integration**: ZeRO-2/ZeRO-3 for memory-efficient distributed training
- 🧠 **Smart Chunking**: Automatic text chunking with overlap for long sequences (32K context)
- 📊 **WandB Logging**: Real-time training metrics and perplexity tracking
- 🎛️ **LoRA Fine-tuning**: Memory-efficient adaptation without full model fine-tuning
- ✅ **EOS Token Handling**: Proper end-of-sequence token injection for correct stopping behavior

## 📋 Requirements

### Hardware
- **Minimum**: 8x GPUs with 80GB+ VRAM (e.g., A100, H100, H200)
- **Recommended**: 8x H200 144GB for 32K context window

### Software
- Python 3.10+
- CUDA 12.0+
- PyTorch 2.1+

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/phased-cpt-finetuning.git
cd phased-cpt-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch first (adjust for your CUDA version)
pip install torch>=2.1.0

# Install Flash Attention (requires torch to be installed)
pip install flash-attn --no-build-isolation

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create JSONL files in the `./dataset` directory with the following format:

```json
{"training_content": "Your training text content here..."}
{"training_content": "Another training sample..."}
```

Required files:
- `dataset/phase1_foundation.jsonl`
- `dataset/phase2_evolution.jsonl`
- `dataset/phase3_pr_mastery.jsonl`

### 3. Configure Training

Edit `config.yaml` to customize:

```yaml
# Model Configuration
model:
  name: "Kwaipilot/KAT-Dev"  # HuggingFace model ID
  size: "32B"

# LoRA Configuration
training:
  lora:
    r: 128          # LoRA rank
    alpha: 256      # LoRA alpha (typically 2*r)
    dropout: 0.05

# Add your WandB API key
wandb:
  api_key: "your-api-key-here"  # Or set WANDB_API_KEY env var
  project: "your-project-name"
```

### 4. Start Training

```bash
# Using the shell script (recommended)
./train_phased.sh

# Or directly with DeepSpeed
deepspeed --num_gpus=8 train_phased.py
```

## 📁 Project Structure

```
.
├── config.yaml              # Main configuration file
├── train_phased.py          # Phased training script
├── train_phased.sh          # Training launcher with pre-flight checks
├── inference.py             # Interactive inference & model comparison
├── requirements.txt         # Python dependencies
├── deepspeed_configs/
│   ├── zero2.json          # DeepSpeed ZeRO-2 config (shorter sequences)
│   └── zero3.json          # DeepSpeed ZeRO-3 config (longer sequences)
└── dataset/                 # Training data directory
    ├── phase1_foundation.jsonl
    ├── phase2_evolution.jsonl
    └── phase3_pr_mastery.jsonl
```

## ⚙️ Configuration Reference

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `micro_batch_size` | 1 | Per-GPU batch size |
| `gradient_accumulation_steps` | 6 | Steps before optimizer update |
| `learning_rate` | 1e-4 | Learning rate |
| `lr_scheduler` | cosine | LR scheduler type |
| `warmup_ratio` | 0.03 | Warmup proportion |
| `weight_decay` | 0.1 | Weight decay |
| `max_grad_norm` | 1.0 | Gradient clipping |

### LoRA Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 128 | LoRA rank (higher = more capacity) |
| `alpha` | 256 | LoRA scaling factor |
| `dropout` | 0.05 | LoRA dropout |
| `target_modules` | q,k,v,o,gate,up,down_proj | Modules to apply LoRA |

### Context & Chunking

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 32768 | Maximum sequence length |
| `overlap_tokens` | 2048 | Overlap between chunks |
| `chunk_long_samples` | true | Enable automatic chunking |

## 🔍 Inference

Compare base model vs fine-tuned outputs interactively:

```bash
python inference.py
```

Features:
- Side-by-side comparison of base vs fine-tuned model
- Adjustable generation parameters (temperature, top_p, etc.)
- Automatic checkpoint discovery
- Token/timing statistics

### Inference Commands

```
/set temperature 0.7    # Adjust temperature
/set max_tokens 2048    # Adjust max generation length
/compare                # Enable comparison mode
/quit                   # Exit
```

## 🧪 Training Tips

### Memory Optimization

1. **32K Context**: Use ZeRO-3 for sequences >16K tokens
2. **Gradient Checkpointing**: Always enable for large models
3. **Sample Packing**: Improves GPU utilization but increases memory

### Hyperparameter Tuning

- **LoRA Rank**: Start with 64, increase to 128 for more capacity
- **Learning Rate**: 1e-4 works well; reduce if loss is unstable
- **Batch Size**: Increase gradient accumulation if OOM

### Monitoring

Training logs to:
- **Console**: Real-time progress
- **WandB**: Loss curves, perplexity, learning rate
- **TensorBoard**: `outputs/<run_name>/tensorboard`

## 📊 Output Structure

After training completes:

```
outputs/
└── phased-kat-dev-lora-YYYYMMDD_HHMMSS/
    ├── phase1_foundation/
    │   └── checkpoint-xxx/
    ├── phase2_evolution/
    │   └── checkpoint-xxx/
    ├── phase3_pr_mastery/
    │   └── checkpoint-xxx/      # Final adapters
    └── tensorboard/
```

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce `micro_batch_size` to 1
- Enable gradient checkpointing
- Switch to ZeRO-3 (`zero3.json`)
- Reduce `max_tokens`

### Slow Training
- Enable `sample_packing: true`
- Use Flash Attention 2
- Check GPU utilization with `nvidia-smi`

### Model Not Stopping Generation
- Ensure you're using this codebase (includes EOS token fix)
- Verify `tokenizer.eos_token` is set correctly

### DeepSpeed Issues
- Update DeepSpeed: `pip install -U deepspeed`
- Check NCCL: `export NCCL_DEBUG=INFO`

## 📝 Citation

If you use this framework, please cite:

```bibtex
@software{phased_cpt_finetuning,
  title = {Phased Continual Pre-Training with LoRA},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/phased-cpt-finetuning}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Note**: This codebase includes a critical fix for EOS token handling that ensures the model learns proper stopping behavior during training.

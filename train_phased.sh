#!/bin/bash

# Color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Read config dynamically
CONFIG=$(python3 << 'PYEOF'
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = config.get('model', {})
training = config.get('training', {})
lora = training.get('lora', {})

print(f"MODEL_NAME={model.get('name', 'Unknown')}")
print(f"MODEL_SIZE={model.get('size', 'Unknown')}")
print(f"MODEL_TYPE={model.get('type', 'base')}")
print(f"LORA_R={lora.get('r', 64)}")
print(f"LORA_ALPHA={lora.get('alpha', 128)}")
PYEOF
)
eval "$CONFIG"

# Detect GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Phased CPT Training${NC}"
echo -e "${CYAN}  Sequential Fine-tuning Across Phases${NC}"
echo -e "${CYAN}============================================${NC}"
echo -e "${GREEN}Model:${NC} ${MODEL_NAME##*/} (${MODEL_TYPE})"
echo -e "${GREEN}Hardware:${NC} ${GPU_COUNT}x ${GPU_NAMES}"
echo -e "${GREEN}Method:${NC} LoRA (Rank ${LORA_R}, Alpha ${LORA_ALPHA})"
echo -e "${CYAN}============================================${NC}"
echo ""

# Pre-flight checks
echo -e "${YELLOW}Running pre-flight checks...${NC}"

# Check datasets
DATASETS=("phase1_foundation.jsonl" "phase2_evolution.jsonl" "phase3_pr_mastery.jsonl")
for ds in "${DATASETS[@]}"; do
    if [ ! -f "dataset/$ds" ]; then
        echo -e "${RED}✗ Error: dataset/$ds not found!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} Found $ds"
done

# Check Python packages
python3 -c "import transformers, peft, trl, deepspeed, wandb" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Error: Required packages not installed!${NC}"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo -e "  ${GREEN}✓${NC} Ready to start phased training"

# Show GPU status
echo ""
echo -e "${BLUE}GPU Status:${NC}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# Show training plan
echo ""
echo -e "${BLUE}Phased Training Plan:${NC}"
python3 << 'PYEOF'
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

phases = config.get('phased_training', {}).get('phases', [])
for i, phase in enumerate(phases, 1):
    # Count samples in dataset
    dataset_file = f"dataset/{phase['dataset_file']}"
    try:
        with open(dataset_file, 'r') as f:
            sample_count = sum(1 for _ in f)
    except:
        sample_count = "?"
    print(f"  Phase {i}: {phase['name']} ({phase['epochs']} epochs) - {sample_count} samples")
PYEOF

# Show configuration
echo ""
echo -e "${BLUE}Training Configuration:${NC}"
python3 << 'PYEOF'
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = config.get('model', {})
training = config.get('training', {})
lora = training.get('lora', {})
dataset = config.get('dataset', {})
phases = config.get('phased_training', {}).get('phases', [])

micro_batch = training.get('micro_batch_size', 1)
grad_accum = training.get('gradient_accumulation_steps', 6)
gpu_count = 8  # Assuming 8 GPUs
effective_batch = micro_batch * grad_accum * gpu_count

total_epochs = sum(p['epochs'] for p in phases)

print(f"  Model: {model.get('name', 'Unknown')}")
print(f"  Batch size: {micro_batch} micro × {grad_accum} grad_accum × {gpu_count} GPUs = {effective_batch} effective")
print(f"  Learning rate: {training.get('learning_rate', 0.0001)}")
print(f"  Total epochs: {total_epochs} ({'+'.join(str(p['epochs']) for p in phases)} across phases)")
print(f"  LoRA: Rank {lora.get('r', 64)}, Alpha {lora.get('alpha', 128)}")
print(f"  Context length: {dataset.get('max_tokens', 16384)} tokens")
print(f"  DeepSpeed: ZeRO-3 (for 16K context)")
PYEOF

# Check wandb
echo ""
echo -e "${BLUE}Experiment Tracking:${NC}"
python3 << 'PYEOF'
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

wandb_config = config.get('wandb', {})
if wandb_config.get('enabled', False):
    try:
        import wandb
        wandb.login(key=wandb_config.get('api_key', ''), relogin=True)
        print("  ✓ Weights & Biases (logged in)")
    except:
        print("  ⚠ Weights & Biases (login failed)")
else:
    print("  ○ Weights & Biases (disabled)")
PYEOF

# Countdown
echo ""
echo -e "${YELLOW}Press Ctrl+C in the next 5 seconds to abort...${NC}"
for i in {5..1}; do
    sleep 1
done

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}Starting Phased Training...${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# Log file
LOG_FILE="train_phased_log.txt"
echo -e "${GREEN}Training logs:${NC} $LOG_FILE"
echo -e "${GREEN}Monitoring:${NC}"
echo "  • Console: Watch output below"
echo "  • Logs: tail -f $LOG_FILE"
echo "  • W&B: Check your wandb dashboard"
echo "  • GPUs: watch -n 1 nvidia-smi"
echo ""
echo -e "${CYAN}============================================${NC}"
echo ""

# Set timestamp as environment variable BEFORE launching DeepSpeed
# This ensures all 8 ranks get the exact same timestamp
export TRAINING_RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Training run timestamp: $TRAINING_RUN_TIMESTAMP"

# Clean up old timestamp file if exists
rm -f /tmp/training_timestamp.txt

# Launch training with DeepSpeed
echo -e "${GREEN}Launching phased training with DeepSpeed ZeRO-3 (${GPU_COUNT} GPUs)...${NC}"
echo ""

deepspeed --num_gpus=$GPU_COUNT train_phased.py 2>&1 | tee $LOG_FILE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  Training Complete!${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo -e "${GREEN}Check outputs/ directory for:${NC}"
    echo "  • Final model adapters"
    echo "  • Phase checkpoints"
    echo "  • Training metrics"
else
    echo ""
    echo -e "${RED}============================================${NC}"
    echo -e "${RED}  Training Failed${NC}"
    echo -e "${RED}============================================${NC}"
    echo ""
    echo "Check $LOG_FILE for error details"
    exit 1
fi

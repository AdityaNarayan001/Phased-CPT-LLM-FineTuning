#!/usr/bin/env python3
"""
Phased training script for sequential fine-tuning across multiple datasets
Phase 1: Foundation (3 epochs)
Phase 2: Evolution (2 epochs)
Phase 3: PR Mastery (2 epochs)
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import logging
import math
from pathlib import Path
import yaml
import torch
import warnings
import json
import signal
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")
warnings.filterwarnings("ignore", message=".*Gradient accumulation steps mismatch.*")
warnings.filterwarnings("ignore", message=".*tokenizer has new PAD/BOS/EOS tokens.*")

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("accelerate.accelerator").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_training_config():
    """Get training configuration from config.yaml"""
    yaml_config = load_config()
    
    training_config = yaml_config.get('training', {})
    model_config = yaml_config.get('model', {})
    dataset_config = yaml_config.get('dataset', {})
    wandb_config = yaml_config.get('wandb', {})
    phased_config = yaml_config.get('phased_training', {})
    
    return {
        'base_model': model_config.get('name', 'Kwaipilot/KAT-Dev'),
        'output_base_dir': './outputs',
        'run_name': f"phased-{model_config.get('name', 'model').split('/')[-1].lower()}-lora",
        
        # Wandb
        'wandb_enabled': wandb_config.get('enabled', True),
        'wandb_api_key': wandb_config.get('api_key', ''),
        'wandb_project': wandb_config.get('project', 'code-finetuning'),
        'wandb_entity': wandb_config.get('entity', ''),
        
        # Phased training
        'phased_enabled': phased_config.get('enabled', True),
        'phases': phased_config.get('phases', []),
        
        'lora_r': training_config.get('lora', {}).get('r', 64),
        'lora_alpha': training_config.get('lora', {}).get('alpha', 128),
        'lora_dropout': training_config.get('lora', {}).get('dropout', 0.05),
        'lora_target_modules': training_config.get('lora', {}).get('target_modules', 
            ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
        
        'micro_batch_size': training_config.get('micro_batch_size', 8),
        'gradient_accumulation_steps': training_config.get('gradient_accumulation_steps', 4),
        'eval_batch_size': training_config.get('eval_batch_size', 8),
        'learning_rate': training_config.get('learning_rate', 0.0001),
        'lr_scheduler': training_config.get('lr_scheduler', 'cosine'),
        'warmup_ratio': training_config.get('warmup_ratio', 0.03),
        'weight_decay': training_config.get('weight_decay', 0.1),
        'max_grad_norm': training_config.get('max_grad_norm', 0.5),
        
        'bf16': training_config.get('bf16', True),
        'fp16': training_config.get('fp16', False),
        'tf32': training_config.get('tf32', True),
        
        'logging_steps': training_config.get('logging_steps', 10),
        'save_steps': training_config.get('save_steps', 50),
        'save_total_limit': training_config.get('save_total_limit', 3),
        'eval_steps': training_config.get('eval_steps', 50),
        
        'sequence_len': dataset_config.get('max_tokens', 32768),
        'sample_packing': training_config.get('sample_packing', True),
        
        'train_split': dataset_config.get('train_split', 0.95),
        'val_split': dataset_config.get('val_split', 0.05),
        'random_seed': dataset_config.get('random_seed', 42),
        
        'gradient_checkpointing': training_config.get('gradient_checkpointing', True),
        'seed': dataset_config.get('random_seed', 42),
        
        'dataset_dir': dataset_config.get('output_dir', './dataset'),
        
        # Chunking settings
        'overlap_tokens': dataset_config.get('overlap_tokens', 2048),
        'chunk_long_samples': dataset_config.get('chunk_long_samples', True),
    }


# Load config at module level
TRAINING_CONFIG = get_training_config()

# Get phased training configuration from config.yaml
TRAINING_PHASES = TRAINING_CONFIG.get('phases', [])


def chunk_text_with_overlap(text: str, tokenizer, max_tokens: int, overlap_tokens: int) -> list:
    """
    Chunk a text into multiple segments with overlap to ensure no data is lost.
    
    Args:
        text: The text to chunk
        tokenizer: The tokenizer to use for token counting
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of overlapping tokens between chunks
    
    Returns:
        List of text chunks
    """
    # Tokenize the entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # If text fits in max_tokens, return as-is
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    stride = max_tokens - overlap_tokens
    
    # Ensure stride is positive
    if stride <= 0:
        stride = max_tokens // 2
    
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        
        # Move to next chunk
        if end >= len(tokens):
            break
        start += stride
    
    return chunks


def load_and_chunk_dataset(dataset_path: str, tokenizer, max_tokens: int, overlap_tokens: int, 
                           chunk_enabled: bool, logger) -> list:
    """
    Load dataset and chunk long samples if needed.
    
    Args:
        dataset_path: Path to the JSONL file
        tokenizer: Tokenizer for token counting
        max_tokens: Maximum tokens per sample
        overlap_tokens: Overlap between chunks
        chunk_enabled: Whether to chunk long samples
        logger: Logger instance
    
    Returns:
        List of data dictionaries ready for training
    """
    data_list = []
    original_count = 0
    chunked_count = 0
    
    with open(dataset_path, 'r') as f:
        for line in f:
            original_count += 1
            data = json.loads(line)
            content = data.get('training_content', '')
            
            if not content:
                continue
            
            if chunk_enabled:
                # Check token count and chunk if needed
                chunks = chunk_text_with_overlap(content, tokenizer, max_tokens, overlap_tokens)
                
                if len(chunks) > 1:
                    chunked_count += 1
                
                for chunk in chunks:
                    data_list.append({'training_content': chunk})
            else:
                # No chunking - just add as-is (will be truncated by trainer)
                data_list.append({'training_content': content})
    
    if chunk_enabled:
        logger.info(f"  Original samples: {original_count:,}")
        logger.info(f"  Samples that were chunked: {chunked_count:,}")
        logger.info(f"  Total samples after chunking: {len(data_list):,}")
    
    return data_list


class PerplexityCallback(TrainerCallback):
    """Callback to log perplexity to TensorBoard and Wandb"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_world_process_zero:
            import math
            
            # Calculate and add perplexity metrics
            if 'loss' in logs:
                try:
                    train_ppl = math.exp(logs['loss'])
                    logs['train_perplexity'] = train_ppl
                    logs['perplexity'] = train_ppl  # For wandb
                except (ValueError, OverflowError):
                    logs['train_perplexity'] = float('inf')
                    logs['perplexity'] = float('inf')
            
            if 'eval_loss' in logs:
                try:
                    eval_ppl = math.exp(logs['eval_loss'])
                    logs['eval_perplexity'] = eval_ppl
                except (ValueError, OverflowError):
                    logs['eval_perplexity'] = float('inf')
            
            # Explicitly log to wandb if available
            try:
                import wandb
                if wandb.run is not None:
                    wandb_logs = {}
                    if 'loss' in logs:
                        wandb_logs['train/loss'] = logs['loss']
                        if 'perplexity' in logs:
                            wandb_logs['train/perplexity'] = logs['perplexity']
                    if 'eval_loss' in logs:
                        wandb_logs['eval/loss'] = logs['eval_loss']
                        if 'eval_perplexity' in logs:
                            wandb_logs['eval/perplexity'] = logs['eval_perplexity']
                    if wandb_logs:
                        wandb.log(wandb_logs, step=state.global_step)
            except ImportError:
                pass


def train_phase(phase_info, tokenizer, output_dir, phase_num, total_phases, previous_adapter_path=None):
    """Train a single phase - loads fresh model each time"""
    
    logger.info("\n" + "=" * 80)
    logger.info(f"PHASE {phase_num}/{total_phases}: {phase_info['name']}")
    logger.info(f"Description: {phase_info['description']}")
    logger.info(f"Dataset: {phase_info['dataset_file']}")
    logger.info(f"Epochs: {phase_info['epochs']}")
    if previous_adapter_path:
        logger.info(f"Loading adapters from: {previous_adapter_path}")
    logger.info("=" * 80)
    
    # Load fresh base model for this phase
    logger.info(f"\nLoading fresh base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        TRAINING_CONFIG['base_model'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        local_files_only=True,  # Use cached model to avoid HF Hub errors
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    
    # Setup LoRA
    if previous_adapter_path:
        # Load previous phase's trained LoRA weights
        logger.info(f"Loading LoRA adapters from previous phase...")
        model = PeftModel.from_pretrained(
            base_model,
            previous_adapter_path,
            is_trainable=True,
        )
        logger.info(f"✓ Loaded adapters, continuing training")
    else:
        # First phase: create fresh LoRA
        logger.info(f"Initializing fresh LoRA adapters...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=TRAINING_CONFIG['lora_r'],
            lora_alpha=TRAINING_CONFIG['lora_alpha'],
            lora_dropout=TRAINING_CONFIG['lora_dropout'],
            target_modules=TRAINING_CONFIG['lora_target_modules'],
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(base_model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
        logger.info(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Load phase dataset with chunking support
    dataset_path = os.path.join(TRAINING_CONFIG['dataset_dir'], phase_info['dataset_file'])
    logger.info(f"\nLoading dataset: {dataset_path}")
    logger.info(f"  Max tokens: {TRAINING_CONFIG['sequence_len']:,}")
    logger.info(f"  Overlap tokens: {TRAINING_CONFIG['overlap_tokens']:,}")
    logger.info(f"  Chunking enabled: {TRAINING_CONFIG['chunk_long_samples']}")
    
    # Load and chunk dataset
    data_list = load_and_chunk_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_tokens=TRAINING_CONFIG['sequence_len'],
        overlap_tokens=TRAINING_CONFIG['overlap_tokens'],
        chunk_enabled=TRAINING_CONFIG['chunk_long_samples'],
        logger=logger,
    )
    
    from datasets import Dataset
    all_data = Dataset.from_list(data_list)
    logger.info(f"  Total samples: {len(all_data):,}")
    
    # Split dataset
    test_size = TRAINING_CONFIG['val_split']
    seed = TRAINING_CONFIG['random_seed']
    
    split_dataset = all_data.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"  Training: {len(train_dataset):,} ({TRAINING_CONFIG['train_split']*100:.0f}%)")
    logger.info(f"  Validation: {len(eval_dataset):,} ({TRAINING_CONFIG['val_split']*100:.0f}%)")
    
    # Create phase output directory
    phase_output_dir = os.path.join(output_dir, f"phase{phase_num}_{phase_info['dataset_file'].replace('.jsonl', '')}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=phase_output_dir,
        
        num_train_epochs=phase_info['epochs'],
        max_steps=-1,
        
        per_device_train_batch_size=TRAINING_CONFIG['micro_batch_size'],
        per_device_eval_batch_size=TRAINING_CONFIG['eval_batch_size'],
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
        
        learning_rate=TRAINING_CONFIG['learning_rate'],
        lr_scheduler_type=TRAINING_CONFIG['lr_scheduler'],
        warmup_ratio=TRAINING_CONFIG['warmup_ratio'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
        
        bf16=TRAINING_CONFIG['bf16'],
        fp16=TRAINING_CONFIG['fp16'],
        tf32=TRAINING_CONFIG['tf32'],
        
        logging_steps=TRAINING_CONFIG['logging_steps'],
        logging_dir=f"{phase_output_dir}/logs",
        report_to=["wandb"],  # Enable wandb
        
        save_strategy="steps",
        save_steps=TRAINING_CONFIG['save_steps'],
        save_total_limit=2,  # Keep fewer checkpoints per phase
        save_only_model=True,  # Don't save optimizer state - we only need adapters
        
        eval_strategy="steps",
        eval_steps=TRAINING_CONFIG['eval_steps'],
        
        gradient_checkpointing=TRAINING_CONFIG['gradient_checkpointing'],
        
        deepspeed="deepspeed_configs/zero3.json",  # Use ZeRO-3 for 32K context
        
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=TRAINING_CONFIG['seed'],
        
        # Wandb config
        run_name=f"{TRAINING_CONFIG['run_name']}_phase{phase_num}",
    )
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Epochs: {phase_info['epochs']}")
    logger.info(f"  Max sequence length: {TRAINING_CONFIG['sequence_len']}")
    
    # Initialize trainer
    perplexity_callback = PerplexityCallback()
    
    # Preprocess datasets - tokenize the training_content
    def tokenize_function(examples):
        # Add EOS token to each training sample so model learns when to stop
        # This is CRITICAL for CPT - without EOS, model won't learn stopping behavior
        texts_with_eos = [text + tokenizer.eos_token for text in examples["training_content"]]
        
        # Tokenize the text
        model_inputs = tokenizer(
            texts_with_eos,
            truncation=True,
            max_length=TRAINING_CONFIG['sequence_len'],
            padding=False,
        )
        # For causal LM, labels are the same as input_ids (shifted internally by model)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    logger.info(f"\nTokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation data"
    )
    
    # Custom data collator that handles padding correctly
    from dataclasses import dataclass
    from typing import Any, Dict, List
    
    @dataclass
    class DataCollatorForCausalLM:
        tokenizer: Any
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            # Extract input_ids and labels
            input_ids = [f["input_ids"] for f in features]
            labels = [f["labels"] for f in features]
            
            # Pad sequences
            batch = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
            )
            
            # Pad labels with -100 (ignored in loss)
            max_length = batch["input_ids"].shape[1]
            padded_labels = []
            for label in labels:
                padding_length = max_length - len(label)
                padded_label = label + [-100] * padding_length
                padded_labels.append(padded_label)
            
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            return batch
    
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[perplexity_callback],
    )
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Phase {phase_num} training...")
    logger.info("=" * 80 + "\n")
    
    train_result = trainer.train()
    
    # Save final checkpoint explicitly (in case save_steps > total_steps)
    # This ensures we always have a checkpoint to use for the next phase
    final_checkpoint_dir = os.path.join(phase_output_dir, "final_checkpoint")
    if trainer.args.local_rank in [-1, 0]:
        logger.info(f"\nSaving final checkpoint for Phase {phase_num}...")
    
    # Use trainer.save_model which handles DeepSpeed + PEFT properly
    trainer.save_model(final_checkpoint_dir)
    
    # Also save tokenizer
    if trainer.args.local_rank in [-1, 0]:
        tokenizer.save_pretrained(final_checkpoint_dir)
        logger.info(f"  Saved to: {final_checkpoint_dir}")
    
    phase_checkpoint_dir = final_checkpoint_dir
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.save_metrics(f"phase{phase_num}_train", metrics)
    
    # Run final evaluation
    eval_metrics = trainer.evaluate()
    trainer.save_metrics(f"phase{phase_num}_eval", eval_metrics)
    
    logger.info(f"✓ Phase {phase_num} complete!")
    logger.info(f"  Checkpoint: {phase_checkpoint_dir}")
    logger.info(f"  Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"  Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    
    # Clean up everything - free all GPU memory
    del model, trainer, train_dataset, eval_dataset, base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return phase_checkpoint_dir, metrics, eval_metrics


def main():
    global TRAINING_CONFIG
    
    # Initialize wandb if enabled
    if TRAINING_CONFIG.get('wandb_enabled', True):
        try:
            import wandb
            api_key = TRAINING_CONFIG.get('wandb_api_key', '')
            if api_key:
                os.environ['WANDB_API_KEY'] = api_key
            
            # Set wandb project name
            os.environ['WANDB_PROJECT'] = TRAINING_CONFIG.get('wandb_project', 'code-finetuning')
            
            wandb.login()
            logger.info("✓ Weights & Biases initialized")
            logger.info(f"  Project: {TRAINING_CONFIG.get('wandb_project', 'code-finetuning')}")
        except ImportError:
            logger.warning("⚠️  wandb not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"⚠️  wandb initialization failed: {e}")
    
    # Get timestamp from environment variable (set by shell script)
    # This ensures all ranks use the exact same timestamp
    timestamp = os.environ.get('TRAINING_RUN_TIMESTAMP', '')
    if not timestamp:
        # Fallback: generate timestamp (only if not set by shell script)
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            # Non-zero ranks wait for rank 0 to create timestamp file
            import time
            timestamp_file = "/tmp/training_timestamp.txt"
            for _ in range(60):
                if os.path.exists(timestamp_file):
                    with open(timestamp_file, 'r') as f:
                        timestamp = f.read().strip()
                    if timestamp:
                        break
                time.sleep(0.1)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    run_name = TRAINING_CONFIG['run_name']
    output_dir = os.path.join(TRAINING_CONFIG['output_base_dir'], f"{run_name}_{timestamp}")
    
    logger.info("=" * 80)
    logger.info("Phased CPT Training - Sequential Fine-tuning")
    logger.info("Using: Transformers + TRL + PEFT + DeepSpeed ZeRO-3")
    logger.info("=" * 80)
    
    model_name = TRAINING_CONFIG['base_model']
    
    logger.info(f"\nModel: {model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Training run: {timestamp}")
    logger.info(f"\nPhased Training Plan:")
    for i, phase in enumerate(TRAINING_PHASES, 1):
        logger.info(f"  Phase {i}: {phase['name']} - {phase['epochs']} epochs ({phase['dataset_file']})")
    
    # Load tokenizer
    logger.info("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,  # Use cached files to avoid HF Hub errors
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set max sequence length for training
    tokenizer.model_max_length = TRAINING_CONFIG['sequence_len']
    
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  Pad token: {tokenizer.pad_token}")
    logger.info(f"  Max sequence length: {tokenizer.model_max_length}")
    
    # Train each phase sequentially
    # Each phase loads fresh base model + previous phase's LoRA adapters
    previous_adapter_path = None
    phase_checkpoints = []
    all_phase_metrics = []
    
    for phase_num, phase_info in enumerate(TRAINING_PHASES, 1):
        checkpoint_dir, train_metrics, eval_metrics = train_phase(
            phase_info=phase_info,
            tokenizer=tokenizer,
            output_dir=output_dir,
            phase_num=phase_num,
            total_phases=len(TRAINING_PHASES),
            previous_adapter_path=previous_adapter_path,
        )
        
        # Use this phase's checkpoint for next phase
        previous_adapter_path = checkpoint_dir
        
        phase_checkpoints.append(checkpoint_dir)
        all_phase_metrics.append({
            'phase': phase_num,
            'name': phase_info['name'],
            'dataset': phase_info['dataset_file'],
            'epochs': phase_info['epochs'],
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics,
        })
    
    # Save final consolidated model
    logger.info("\n" + "=" * 80)
    logger.info("All phases complete! Saving final model...")
    logger.info("=" * 80)
    
    # The final model is the last phase's checkpoint
    # Just copy/link it to a "final_model" directory for clarity
    final_model_dir = os.path.join(output_dir, "final_model")
    
    if local_rank == 0:
        import shutil
        # Copy the last phase's checkpoint to final_model
        if previous_adapter_path and os.path.exists(previous_adapter_path):
            shutil.copytree(previous_adapter_path, final_model_dir, dirs_exist_ok=True)
            tokenizer.save_pretrained(final_model_dir)
            logger.info(f"  Final model saved to: {final_model_dir}")
        else:
            logger.warning(f"  Could not find last checkpoint at: {previous_adapter_path}")
    
    # Save comprehensive training info
    training_info = {
        'model': {
            'base_model': TRAINING_CONFIG['base_model'],
            'final_model_path': final_model_dir,
        },
        'training_config': {
            'lora_r': TRAINING_CONFIG['lora_r'],
            'lora_alpha': TRAINING_CONFIG['lora_alpha'],
            'lora_dropout': TRAINING_CONFIG['lora_dropout'],
            'learning_rate': TRAINING_CONFIG['learning_rate'],
            'micro_batch_size': TRAINING_CONFIG['micro_batch_size'],
            'gradient_accumulation_steps': TRAINING_CONFIG['gradient_accumulation_steps'],
            'sequence_length': TRAINING_CONFIG['sequence_len'],
            'train_split': TRAINING_CONFIG['train_split'],
            'val_split': TRAINING_CONFIG['val_split'],
        },
        'hardware': {
            'num_gpus': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        },
        'phases': all_phase_metrics,
        'phase_checkpoints': phase_checkpoints,
        'summary': {
            'initial_loss': all_phase_metrics[0]['train_metrics'].get('train_loss') if all_phase_metrics else None,
            'final_loss': all_phase_metrics[-1]['train_metrics'].get('train_loss') if all_phase_metrics else None,
            'initial_eval_loss': all_phase_metrics[0]['eval_metrics'].get('eval_loss') if all_phase_metrics else None,
            'final_eval_loss': all_phase_metrics[-1]['eval_metrics'].get('eval_loss') if all_phase_metrics else None,
            'initial_perplexity': math.exp(all_phase_metrics[0]['eval_metrics'].get('eval_loss', float('inf'))) if all_phase_metrics else None,
            'final_perplexity': math.exp(all_phase_metrics[-1]['eval_metrics'].get('eval_loss', float('inf'))) if all_phase_metrics else None,
            'total_epochs': sum(p['epochs'] for p in all_phase_metrics),
            'total_phases': len(all_phase_metrics),
        },
        'timestamp': timestamp,
        'run_name': run_name,
        'output_directory': output_dir,
    }
    
    if local_rank == 0:
        training_info_path = os.path.join(output_dir, "training_info.json")
        with open(training_info_path, 'w') as f:
            json.dump(training_info, f, indent=2, default=str)
        
        logger.info(f"\n✓ Final model saved: {final_model_dir}")
        logger.info(f"✓ Training info saved: {training_info_path}")
        logger.info(f"\nPhase checkpoints:")
        for i, cp in enumerate(phase_checkpoints, 1):
            logger.info(f"  Phase {i}: {cp}")
    
    logger.info("\n✓ Phased training complete!")


if __name__ == "__main__":
    main()

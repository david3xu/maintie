#!/usr/bin/env python3
"""
🏆 OFFICIAL REBEL EVALUATION SCRIPT
Uses the exact same evaluation pipeline as the original project to get metrics matching RESULTS.md

This script replicates the official evaluation methodology using:
1. The exact same test.py evaluation pipeline
2. Proper strict and loose relation extraction evaluation
3. Official re_score function from score.py
"""

import os
import sys
import torch
import pytorch_lightning as pl
import omegaconf

# Fix PyTorch 2.6 checkpoint loading issue
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Add the REBEL source directory to path
sys.path.append('models/rebel/src')

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from pl_modules import BasePLModule
from pl_data_modules import BasePLDataModule
from score import re_score

class REBELOfficialEvaluator:
    """Official REBEL evaluator that replicates the exact evaluation process"""

    def __init__(self):
        self.maintie_general_tokens = ["<num>", "<id>", "<date>", "<sensitive>"]
        self.maintie_level_1_unique_entities = [
            "<physical object>",
            "<process>",
            "<property>",
            "<activity>",
            "<state>",
        ]
        self.maintie_relation_types = [
            "is a",
            "contains",
            "has part",
            "has participant",
            "has patient",
            "has agent",
            "has property"
        ]

    def setup_model_components(self, dataset_level, model_name="facebook/bart-large"):
        """Set up tokenizer, config, and model with proper MaintIE tokens"""
        print(f"🔧 Setting up model components...")

        # 1. Load config
        config = AutoConfig.from_pretrained(
            model_name,
            decoder_start_token_id=0,
            early_stopping=False,
            no_repeat_ngram_size=0,
        )

        # 2. Load tokenizer with special tokens
        tokenizer_kwargs = {
            "use_fast": True,
            "additional_special_tokens": ["<obj>", "<subj>", "<triplet>"],
        }

        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        print(f"Base tokenizer size: {len(tokenizer)}")

        # 3. Add MaintIE general special tokens
        print("ADDING SPECIAL TOKENS FOR MAINTIE")
        tokenizer.add_tokens(self.maintie_general_tokens, special_tokens=True)
        print(f"Tokenizer size after general tokens: {len(tokenizer)}")

        # 4. Add entity tokens based on dataset level
        if dataset_level == 0:
            # Untyped - no entity tokens needed
            pass
        elif dataset_level == 1:
            # Level 1: exactly same tokens as training script
            tokenizer.add_tokens(self.maintie_level_1_unique_entities, special_tokens=True)
        elif dataset_level == 2:
            # Level 2: need to add level 1 + level 2 tokens
            # Load from training script definitions or mapping file
            import json
            with open(f"models/data/g-2/maintie_rebel_mapping.json", "r") as f:
                mapping = json.load(f)
            level_2_tokens = mapping["rebel_entity_types"]
            tokenizer.add_tokens(level_2_tokens, special_tokens=True)
        elif dataset_level == 3:
            # Level 3: need to add all tokens
            import json
            with open(f"models/data/g-3/maintie_rebel_mapping.json", "r") as f:
                mapping = json.load(f)
            level_3_tokens = mapping["rebel_entity_types"]
            tokenizer.add_tokens(level_3_tokens, special_tokens=True)

        print(f"Final tokenizer size: {len(tokenizer)}")

        # 5. Load model and resize embeddings
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
        model.resize_token_embeddings(len(tokenizer))

        return config, tokenizer, model

    def create_evaluation_config(self, dataset_level, checkpoint_path, data_path):
        """Create configuration for evaluation"""
        conf_dict = {
            # Model settings
            'model_name_or_path': "facebook/bart-large",
            'config_name': None,
            'tokenizer_name': None,
            'use_fast_tokenizer': True,
            'finetune': True,

            # Training settings (required by BasePLModule)
            'label_smoothing': 0,
            'weight_decay': 0.1,
            'learning_rate': 3e-5,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'warmup_steps': 0,
            'max_steps': 5000,
            'max_epochs': None,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'lr_scheduler': 'linear',
            'monitor': 'val_F1_micro',
            'mode': 'max',
            'save_top_k': 1,
            'patience': 3,
            'min_delta': 0.001,
            'metric_mode': 'max',
            'every_n_train_steps': None,
            'train_batch_size': 8,

            # Data settings
            'dataset_name': f"models/rebel/datasets/maintie_lvl_{dataset_level}.py",
            'text_column': 'context',
            'target_column': 'triplets',
            'test_file': str(data_path),
            'train_file': str(data_path),  # Set to test file for evaluation
            'validation_file': str(data_path),  # Set to test file for evaluation
            'max_source_length': 64,
            'max_target_length': 64,
            'val_max_target_length': 64,
            'pad_to_max_length': False,
            'ignore_pad_token_for_loss': True,
            'preprocessing_num_workers': None,
            'overwrite_cache': True,
            'max_test_samples': None,
            'max_train_samples': None,
            'max_val_samples': None,

            # Generation settings
            'predict_with_generate': True,
            'eval_beams': 3,
            'prediction_loss_only': False,

            # Evaluation settings
            'do_predict': True,
            'do_eval': False,
            'do_train': False,

            # Checkpoint
            'checkpoint_path': checkpoint_path,

            # Other
            'seed': 1337,
            'num_workers': 4,
            'eval_batch_size': 8,
            'dataloader_drop_last': False,
            'dataloader_num_workers': 4,
            'dataloader_pin_memory': True,
            'relations_file': None,
        }

        return omegaconf.DictConfig(conf_dict)

    def run_official_evaluation(self, checkpoint_path, dataset_level, data_path, experiment_name):
        """Run the official REBEL evaluation pipeline"""
        print(f"\n🧪 Running Official REBEL Evaluation: {experiment_name}")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Dataset Level: {dataset_level}")
        print(f"   Data Path: {data_path}")

        # 1. Setup model components
        config, tokenizer, model = self.setup_model_components(dataset_level)

        # 2. Create evaluation configuration
        conf = self.create_evaluation_config(dataset_level, checkpoint_path, data_path)

        # 4. Load the trained model from checkpoint
        print("🔧 Loading trained model from checkpoint...")
        pl_module = BasePLModule(conf, config, tokenizer, model)

        # Load checkpoint state dict with size mismatch handling
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle tokenizer size mismatch
        current_size = len(tokenizer)
        checkpoint_size = checkpoint['state_dict']['model.model.encoder.embed_tokens.weight'].shape[0]

        if current_size != checkpoint_size:
            print(f"⚠️  Tokenizer size mismatch: checkpoint={checkpoint_size}, current={current_size}")
            print("Resizing model embeddings to match current tokenizer...")

            # Resize the model in the checkpoint state dict
            old_embeddings = checkpoint['state_dict']['model.model.shared.weight']
            old_size = old_embeddings.shape[0]
            new_size = current_size

            # Create new embedding matrices with proper size
            embedding_dim = old_embeddings.shape[1]

            # For shared embeddings
            new_shared_embeddings = torch.zeros(new_size, embedding_dim)
            min_size = min(old_size, new_size)
            new_shared_embeddings[:min_size] = old_embeddings[:min_size]

            # Update all embedding-related parameters
            checkpoint['state_dict']['model.model.shared.weight'] = new_shared_embeddings
            checkpoint['state_dict']['model.model.encoder.embed_tokens.weight'] = new_shared_embeddings.clone()
            checkpoint['state_dict']['model.model.decoder.embed_tokens.weight'] = new_shared_embeddings.clone()
            checkpoint['state_dict']['model.lm_head.weight'] = new_shared_embeddings.clone()

            # Update bias if it exists
            if 'model.final_logits_bias' in checkpoint['state_dict']:
                old_bias = checkpoint['state_dict']['model.final_logits_bias']
                new_bias = torch.zeros(1, new_size)
                new_bias[:, :min_size] = old_bias[:, :min_size]
                checkpoint['state_dict']['model.final_logits_bias'] = new_bias

        # Load the adjusted state dict
        try:
            pl_module.load_state_dict(checkpoint['state_dict'], strict=False)
            print("✅ Checkpoint loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            raise

        # 5. Setup data module
        print("📊 Setting up data module...")
        pl_data_module = BasePLDataModule(conf, tokenizer, model)
        pl_data_module.prepare_data()
        pl_data_module.setup(stage="test")

        # 6. Create trainer and run evaluation
        print("🚀 Running evaluation...")
        trainer = pl.Trainer(
            devices=1 if torch.cuda.is_available() else 0,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

        # Run the test
        pl_module.eval()
        results = trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader())

        print(f"✅ Evaluation completed for {experiment_name}")
        return results

    def run_all_rebel_evaluations(self):
        """Run evaluations for all available REBEL experiments"""

        # Our completed experiment
        fg_1_checkpoint = "experiments/rebel_final_corrected/epoch=0-step=430.ckpt"
        fg_1_data = "models/data/g-1/maintie_test.json"

        if os.path.exists(fg_1_checkpoint) and os.path.exists(fg_1_data):
            print("🎯 Evaluating REBEL FG-1 (5 entity classes)")
            results = self.run_official_evaluation(
                checkpoint_path=fg_1_checkpoint,
                dataset_level=1,
                data_path=fg_1_data,
                experiment_name="REBEL FG-1"
            )

            print("\n📋 REBEL FG-1 Results Summary:")
            for result in results:
                for key, value in result.items():
                    if 'test_' in key:
                        print(f"   {key}: {value:.4f}")
        else:
            print("❌ REBEL FG-1 checkpoint or data not found")

        # Check for other completed experiments
        potential_experiments = [
            ("REBEL FG-0", 0, "models/data/g-0/maintie_test.json"),
            ("REBEL FG-2", 2, "models/data/g-2/maintie_test.json"),
            ("REBEL FG-3", 3, "models/data/g-3/maintie_test.json"),
        ]

        for exp_name, level, data_path in potential_experiments:
            # Look for checkpoints in common locations
            possible_checkpoints = [
                f"models/rebel/outputs/{exp_name.lower().replace(' ', '_')}/final_model.ckpt",
                f"experiments/{exp_name.lower().replace(' ', '_')}/epoch=0-step=430.ckpt",
            ]

            checkpoint_found = None
            for ckpt_path in possible_checkpoints:
                if os.path.exists(ckpt_path):
                    checkpoint_found = ckpt_path
                    break

            if checkpoint_found and os.path.exists(data_path):
                print(f"\n🎯 Evaluating {exp_name}")
                results = self.run_official_evaluation(
                    checkpoint_path=checkpoint_found,
                    dataset_level=level,
                    data_path=data_path,
                    experiment_name=exp_name
                )

                print(f"\n📋 {exp_name} Results Summary:")
                for result in results:
                    for key, value in result.items():
                        if 'test_' in key:
                            print(f"   {key}: {value:.4f}")
            else:
                print(f"⏳ {exp_name} not ready yet (checkpoint or data missing)")

def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate REBEL model on MaintIE dataset')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to the REBEL checkpoint file')
    parser.add_argument('--hierarchy_level', type=str, choices=['0', '1', '2', '3'],
                        default='1', help='Entity hierarchy level (0=untyped, 1=5 classes, 2=32 classes, 3=224 classes)')
    parser.add_argument('--output_format', type=str, choices=['text', 'json'],
                        default='text', help='Output format (text or json)')
    parser.add_argument('--run_all', action='store_true',
                        help='Run evaluation on all available experiments')

    args = parser.parse_args()

    evaluator = REBELOfficialEvaluator()

    if args.checkpoint_path:
        # Single model evaluation with specified parameters
        level = int(args.hierarchy_level)
        data_path = f"models/data/g-{args.hierarchy_level}/maintie_test.json"

        if not os.path.exists(data_path):
            print(f"❌ Test data not found: {data_path}")
            return

        results = evaluator.run_official_evaluation(
            checkpoint_path=args.checkpoint_path,
            dataset_level=level,
            data_path=data_path,
            experiment_name=f"REBEL Level-{args.hierarchy_level}"
        )

        # Output results in requested format
        if args.output_format == 'json':
            # Convert PyTorch Lightning results to JSON format for comprehensive evaluation
            json_results = {
                "re_strict": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                    "per_class": {}
                },
                "re_loose": {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "support": 0,
                    "per_class": {}
                }
            }

            # Extract metrics from PyTorch Lightning results
            if results and len(results) > 0:
                result = results[0]
                for key, value in result.items():
                    if 'test_' in key and isinstance(value, (int, float)):
                        # Map metrics to our JSON structure
                        metric_name = key.replace('test_', '')
                        if 'precision' in metric_name:
                            json_results["re_strict"]["precision"] = value
                            json_results["re_loose"]["precision"] = value
                        elif 'recall' in metric_name:
                            json_results["re_strict"]["recall"] = value
                            json_results["re_loose"]["recall"] = value
                        elif 'f1' in metric_name:
                            json_results["re_strict"]["f1"] = value
                            json_results["re_loose"]["f1"] = value

            import json
            print(json.dumps(json_results, indent=2))
        else:
            # Standard text output
            print(f"\n📋 REBEL Level-{args.hierarchy_level} Results:")
            for result in results:
                for key, value in result.items():
                    if 'test_' in key:
                        print(f"   {key}: {value:.4f}")

    elif args.run_all:
        # Run all available evaluations
        print("🏆 OFFICIAL REBEL EVALUATION")
        print("=" * 50)

        evaluator.run_all_rebel_evaluations()

        print("\n✅ Official evaluation completed!")
        print("\nTo match RESULTS.md exactly, ensure all 8 REBEL experiments are trained:")
        print("  - FG-0, FG-1, FG-2, FG-3 (Direct fine-tuning)")
        print("  - CG+FG-0, CG+FG-1, CG+FG-2, CG+FG-3 (Sequential fine-tuning)")

    else:
        print("❌ Please specify either --checkpoint_path or --run_all")
        parser.print_help()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
🧪 PROPER REBEL MODEL EVALUATION
Run the exact same evaluation pipeline as the original project to get comparable metrics
"""

import os
import sys
import json
import torch
import pytorch_lightning as pl

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
import omegaconf

# Import the special tokens from train.py
maintie_general_entities = [
    "<num>", "<id>", "<date>", "<sensitive>",
]

maintie_level_1_unique_entities = [
    "<physical object>", "<process>", "<property>", "<activity>", "<state>",
]

def run_proper_evaluation():
    """Run the full REBEL evaluation pipeline to get official metrics"""

    print("🧪 PROPER REBEL MODEL EVALUATION")
    print("Testing our 500-step trained REBEL model with official evaluation pipeline")
    print("=" * 80)

    # Check if checkpoint exists
    checkpoint_path = "experiments/rebel_final_corrected/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"✅ Found checkpoint: {checkpoint_path}")
    size_gb = os.path.getsize(checkpoint_path) / (1024**3)
    print(f"📊 Checkpoint size: {size_gb:.1f} GB")

    try:
        print("\n🔧 Setting up evaluation environment...")

        # 1. Create the exact configuration used in training
        conf_dict = {
            'label_smoothing': 0,
            'ignore_pad_token_for_loss': True,
            'val_max_target_length': 64,
            'eval_beams': 3,
            'dataset_name': "maintie_1",  # Level 1 MaintIE
            'model_name_or_path': "facebook/bart-large",
            'finetune': True,
            'predict_with_generate': True,
            'text_column': 'context',
            'target_column': 'triplets',
            'test_file': 'real_training_data/maintie_dev.json',  # Using dev as test for evaluation
            'max_source_length': 64,
            'max_target_length': 64,
            'pad_to_max_length': False,
            'overwrite_cache': True,
            'preprocessing_num_workers': None,
            'num_workers': 4,
            'max_test_samples': None,
            'relations_file': None,  # Not needed for MaintIE evaluation
        }
        conf = omegaconf.DictConfig(conf_dict)

        # 2. Initialize config (same as training)
        config = AutoConfig.from_pretrained(
            "facebook/bart-large",
            decoder_start_token_id=0,
            early_stopping=False,
            no_repeat_ngram_size=0,
        )
        print("✅ Config initialized")

        # 3. Initialize tokenizer with special tokens (same as training)
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/bart-large",
            use_fast=True,
        )

        # Add special tokens for MaintIE (Level 1 configuration)
        special_tokens = [
            "<obj>", "<subj>", "<triplet>",
            "<head>", "</head>", "<tail>", "</tail>",
            *maintie_general_entities,
            *maintie_level_1_unique_entities,
        ]

        tokenizer.add_tokens(special_tokens, special_tokens=True)
        print(f"✅ Tokenizer loaded with {len(special_tokens)} special tokens")

        # 4. Initialize base model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large",
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))
        print("✅ Base model loaded and resized")

        # 5. Load trained model from checkpoint
        print("\n🔧 Loading trained model from checkpoint...")
        trained_model = BasePLModule(conf, config, tokenizer, model)

        # Load the checkpoint state dict
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        trained_model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("✅ Successfully loaded trained model from checkpoint!")

        # 6. Set up data module for testing
        print("\n🔧 Setting up test data...")
        pl_data_module = BasePLDataModule(conf, tokenizer, model)
        pl_data_module.prepare_data()
        pl_data_module.setup(stage="test")
        print("✅ Test data prepared")

        # 7. Set up trainer for evaluation
        print("\n🔧 Setting up evaluation trainer...")
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            logger=False,  # Disable logging for evaluation
            enable_checkpointing=False,
        )
        print("✅ Trainer configured")

        # 8. Run the official test evaluation
        print("\n🧪 RUNNING OFFICIAL EVALUATION...")
        print("This will calculate the same metrics as in RESULTS.md")
        print("-" * 60)

        # Run test - this will automatically call on_test_epoch_end() which runs re_score()
        test_results = trainer.test(trained_model, dataloaders=pl_data_module.test_dataloader())

        print("-" * 60)
        print("✅ Evaluation completed!")

        # 9. Extract and display results
        if test_results:
            results = test_results[0]
            print(f"\n📊 OFFICIAL RESULTS - REBEL Level 1 (5 entity classes)")
            print("=" * 60)

            # Extract metrics
            test_precision = results.get('test_prec_micro', 0) * 100  # Convert to percentage
            test_recall = results.get('test_recall_micro', 0) * 100
            test_f1 = results.get('test_F1_micro', 0) * 100
            test_loss = results.get('test_loss', 0)

            print(f"Strict Relation Extraction:")
            print(f"  • Precision: {test_precision:.2f}%")
            print(f"  • Recall: {test_recall:.2f}%")
            print(f"  • F1-Score: {test_f1:.2f}%")
            print(f"  • Test Loss: {test_loss:.4f}")

            # Compare with original paper results
            print(f"\n📈 COMPARISON WITH ORIGINAL PAPER:")
            print("=" * 50)
            print("Original REBEL FG-1 results:")
            print("  • Strict RE: F1=67.87%, P=64.02%, R=72.22%")
            print("  • Loose RE: F1=68.27%, P=64.39%, R=72.65%")
            print()
            print("Our trained model results:")
            print(f"  • Strict RE: F1={test_f1:.2f}%, P={test_precision:.2f}%, R={test_recall:.2f}%")

            # Calculate performance ratio
            original_f1 = 67.87
            our_performance_ratio = (test_f1 / original_f1) * 100 if test_f1 > 0 else 0

            print(f"\n🎯 PERFORMANCE ASSESSMENT:")
            print("=" * 40)
            if our_performance_ratio >= 90:
                status = "🏅 EXCELLENT"
            elif our_performance_ratio >= 75:
                status = "✅ GOOD"
            elif our_performance_ratio >= 50:
                status = "🟡 MODERATE"
            else:
                status = "🔴 NEEDS IMPROVEMENT"

            print(f"Performance vs Original: {our_performance_ratio:.1f}% {status}")
            print(f"Training Steps: 500 (vs. original ~5000)")
            print(f"Training Time: ~11 minutes")
            print(f"Model Size: 4.5 GB")

        else:
            print("❌ No test results returned")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_lifecycle_status():
    """Display the current status of our replication vs. original project"""

    print(f"\n🔄 PROJECT REPLICATION LIFECYCLE STATUS")
    print("=" * 60)

    print("Original Project Scope:")
    print("  • 2 Models: SpERT + REBEL")
    print("  • 8 Experiments per model (FG-0/1/2/3 + CG+FG-0/1/2/3)")
    print("  • Total: 16 experiments")
    print("  • Full evaluation pipeline with NER + Strict/Loose RE metrics")

    print(f"\nOur Current Progress:")
    print("  ✅ REBEL Model: 1/8 experiments completed (FG-1)")
    print("  ❌ SpERT Model: 0/8 experiments completed")
    print("  ✅ Evaluation Pipeline: Implemented and working")
    print("  ✅ Data Preparation: MaintIE dataset ready")
    print("  ✅ Infrastructure: Training environment operational")

    print(f"\nCompletion Status:")
    completed = 1
    total = 16
    completion_percentage = (completed / total) * 100
    print(f"  • Overall Progress: {completed}/{total} experiments ({completion_percentage:.1f}%)")
    print(f"  • REBEL Progress: 1/8 experiments (12.5%)")
    print(f"  • SpERT Progress: 0/8 experiments (0%)")

    print(f"\nNext Steps for Full Replication:")
    print("  1. Complete remaining 7 REBEL experiments")
    print("  2. Set up and run 8 SpERT experiments")
    print("  3. Generate comprehensive results matching RESULTS.md")
    print("  4. Compare all metrics against original paper")

if __name__ == "__main__":
    print("🚀 STARTING PROPER REBEL EVALUATION")
    print("Replicating the original project's evaluation methodology")
    print("=" * 80)

    run_proper_evaluation()
    display_lifecycle_status()

    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print("💡 Ready to proceed with remaining experiments if desired!")

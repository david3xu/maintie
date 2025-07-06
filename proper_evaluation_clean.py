#!/usr/bin/env python3
"""
🧪 PROPER REBEL MODEL EVALUATION - CLEAN VERSION
Run the exact same evaluation pipeline as the original project to get comparable metrics
"""

import os
import sys
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

# Special tokens for MaintIE Level 1
maintie_general_entities = ["<num>", "<id>", "<date>", "<sensitive>"]
maintie_level_1_unique_entities = ["<physical object>", "<process>", "<property>", "<activity>", "<state>"]

def run_evaluation():
    """Run the official REBEL evaluation pipeline"""

    print("🧪 PROPER REBEL MODEL EVALUATION")
    print("=" * 80)

    checkpoint_path = "experiments/rebel_final_corrected/last.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    print(f"✅ Found checkpoint: {checkpoint_path}")

    try:
        # Configuration for evaluation
        conf_dict = {
            'label_smoothing': 0,
            'ignore_pad_token_for_loss': True,
            'val_max_target_length': 64,
            'eval_beams': 3,
            'dataset_name': "models/rebel/datasets/maintie_lvl_1.py",
            'model_name_or_path': "facebook/bart-large",
            'finetune': True,
            'predict_with_generate': True,
            'text_column': 'context',
            'target_column': 'triplets',
            'train_file': 'real_training_data/maintie_train.json',
            'validation_file': 'real_training_data/maintie_dev.json',
            'test_file': 'real_training_data/maintie_dev.json',
            'max_source_length': 64,
            'max_target_length': 64,
            'pad_to_max_length': False,
            'overwrite_cache': True,
            'preprocessing_num_workers': None,
            'num_workers': 4,
            'max_train_samples': None,
            'max_val_samples': None,
            'max_test_samples': None,
            'num_beams': None,
            'source_prefix': None,
            'relations_file': None,
        }
        conf = omegaconf.DictConfig(conf_dict)

        # Initialize model components
        config = AutoConfig.from_pretrained(
            "facebook/bart-large",
            decoder_start_token_id=0,
            early_stopping=False,
            no_repeat_ngram_size=0,
        )

        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", use_fast=True)

        special_tokens = [
            "<obj>", "<subj>", "<triplet>", "<head>", "</head>", "<tail>", "</tail>",
            *maintie_general_entities, *maintie_level_1_unique_entities,
        ]
        tokenizer.add_tokens(special_tokens, special_tokens=True)

        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large", config=config)
        model.resize_token_embeddings(len(tokenizer))

        print("✅ Model components initialized")

        # Load trained model
        trained_model = BasePLModule(conf, config, tokenizer, model)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        trained_model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("✅ Trained model loaded")

        # Set up data and trainer
        pl_data_module = BasePLDataModule(conf, tokenizer, model)
        pl_data_module.prepare_data()
        pl_data_module.setup(stage="test")

        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            logger=False,
            enable_checkpointing=False,
        )

        print("✅ Setup complete")

        # Run evaluation
        print("\n🧪 Running evaluation...")
        test_results = trainer.test(trained_model, dataloaders=pl_data_module.test_dataloader())

        # Display results
        if test_results:
            results = test_results[0]
            precision = results.get('test_prec_micro', 0) * 100
            recall = results.get('test_recall_micro', 0) * 100
            f1 = results.get('test_F1_micro', 0) * 100

            print(f"\n📊 RESULTS - REBEL Level 1 (5 entity classes)")
            print("=" * 60)
            print(f"Strict Relation Extraction:")
            print(f"  • F1-Score: {f1:.2f}%")
            print(f"  • Precision: {precision:.2f}%")
            print(f"  • Recall: {recall:.2f}%")

            print(f"\n📈 vs. Original Paper (REBEL FG-1):")
            print("  • Original: F1=67.87%, P=64.02%, R=72.22%")
            print(f"  • Our Model: F1={f1:.2f}%, P={precision:.2f}%, R={recall:.2f}%")

            performance_ratio = (f1 / 67.87) * 100 if f1 > 0 else 0
            print(f"  • Performance: {performance_ratio:.1f}% of original")

        else:
            print("❌ No results returned")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def show_project_status():
    """Show replication status vs original project"""
    print(f"\n🔄 PROJECT LIFECYCLE STATUS")
    print("=" * 50)
    print("Original Project: 2 models × 8 experiments = 16 total")
    print("Our Progress: 1/16 experiments completed (6.2%)")
    print("\nCompleted:")
    print("  ✅ REBEL FG-1: Level 1, 5 entity classes, direct fine-tuning")
    print("\nRemaining:")
    print("  ❌ REBEL: 7 more experiments (FG-0/2/3, CG+FG-0/1/2/3)")
    print("  ❌ SpERT: 8 experiments (all levels)")

if __name__ == "__main__":
    run_evaluation()
    show_project_status()
    print("\n✨ Ready for next experiments!")

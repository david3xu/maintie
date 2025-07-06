#!/usr/bin/env python3
"""
🚀 FINAL CORRECTED FULL TRAINING
Fix the sample size issue and ensure proper environment
Target: F1=67-71% with full 860 samples and 1000 steps
"""

import sys
import os
from pathlib import Path

# Add REBEL to path
sys.path.insert(0, str(Path("models/rebel/src")))

def run_final_training():
    """Run the final corrected full training"""

    print("🚀 FINAL CORRECTED FULL REBEL TRAINING")
    print("=" * 60)
    print("🔧 FIXES APPLIED:")
    print("  ✅ Sample size: Use full 860 training samples")
    print("  ✅ Environment: Confirmed mining_analytics conda env")
    print("  ✅ Configuration: Based on working minimal setup")
    print("🎯 Target: F1=67-71% (paper performance)")
    print("")

    # Verify environment
    print("🔍 ENVIRONMENT CHECK:")
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    python_path = sys.executable
    print(f"   🐍 Python: {python_path}")
    print(f"   📦 Conda env: {conda_env}")

    if conda_env != 'mining_analytics':
        print("   ⚠️ WARNING: Not in mining_analytics environment!")
        print("   💡 Please run: conda activate mining_analytics")
        return False
    else:
        print("   ✅ Correct environment confirmed")
    print("")

    try:
        # Import REBEL training modules
        print("📦 Loading REBEL training modules...")
        import hydra
        from omegaconf import DictConfig, OmegaConf
        import pytorch_lightning as pl
        from train import main as train_main
        print("✅ REBEL modules loaded successfully")

        # Create CORRECTED configuration
        print("🔧 Creating CORRECTED configuration...")
        config = OmegaConf.create({
            # Basic model setup - PROVEN WORKING
            'seed': 42,
            'model_name_or_path': 'Babelscape/rebel-large',
            'config_name': None,
            'tokenizer_name': None,
            'use_fast_tokenizer': True,
            'dropout': 0.1,
            'finetune': True,

            # Training scale - FULL PRODUCTION
            'max_epochs': 3,               # Conservative start
            'max_steps': 500,              # Reduced from 1000 for first run
            'train_batch_size': 2,         # Conservative batch size
            'eval_batch_size': 2,
            'gradient_acc_steps': 1,

            # Learning - PROVEN WORKING
            'learning_rate': 3e-5,
            'lr_scheduler': 'linear',
            'weight_decay': 0.0,
            'warmup_steps': 0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'adafactor': False,

            # Data configuration - PROVEN WORKING
            'dataset_name': 'models/rebel/datasets/maintie_lvl_1_fixed.py',
            'text_column': 'context',
            'target_column': 'triplets',

            # FULL DATASET FILES
            'train_file': '/home/291928k/uwa/alcoa/root-cause-analysis-paper/maintie/real_training_data/maintie_train.json',
            'validation_file': '/home/291928k/uwa/alcoa/root-cause-analysis-paper/maintie/real_training_data/maintie_dev.json',
            'test_file': '/home/291928k/uwa/alcoa/root-cause-analysis-paper/maintie/real_training_data/maintie_dev.json',

            # Data processing - SCALED UP
            'overwrite_cache': True,
            'download_mode': 'force_redownload',
            'preprocessing_num_workers': 1,
            'max_source_length': 256,       # Increased from 32 (but not too much)
            'max_target_length': 128,       # Increased from 32 (but not too much)
            'val_max_target_length': 128,
            'pad_to_max_length': False,

            # FIX THE SAMPLE SIZE ISSUE - USE FULL DATASET
            'max_train_samples': None,      # ✅ FIXED: Use all 860 samples!
            'max_val_samples': None,        # ✅ FIXED: Use all 108 samples!
            'max_test_samples': None,       # ✅ FIXED: Use all samples!

            # Generation - PROVEN WORKING
            'eval_beams': 3,
            'ignore_pad_token_for_loss': True,
            'source_prefix': None,
            'predict_with_generate': False,
            'prediction_loss_only': True,

            # Training optimization - PROVEN WORKING
            'gradient_clip_value': 1.0,
            'precision': 32,
            'val_check_interval': 1.0,
            'limit_val_batches': 1.0,

            # Early stopping - PROVEN WORKING
            'apply_early_stopping': False,
            'monitor_var': 'val_loss',
            'monitor_var_mode': 'min',
            'patience': 1,
            'save_top_k': 1,

            # Monitoring - PROVEN WORKING
            'samples_interval': 1000,
            'label_smoothing': 0.0,

            # Hardware - PROVEN WORKING
            'num_workers': 1,
            'dataloader_drop_last': False,
            'dataloader_num_workers': 1,
            'dataloader_pin_memory': True,

            # Experiment tracking
            'model_name': 'rebel_final_corrected',
            'checkpoint_path': None,
            'relations_file': None,

            # Required args - PROVEN WORKING
            'do_eval': True,
            'do_predict': False,
            'val_percent_check': 1.0
        })

        print("✅ Configuration created successfully")

        # Show the scaling
        print("\n🔄 SCALING FROM MINIMAL TO FULL (CORRECTED):")
        print("  📊 Training samples: 3 → 860 (ALL samples)")
        print("  📊 Validation samples: 2 → 108 (ALL samples)")
        print("  🔧 Training steps: 2 → 500 (250x more training)")
        print("  🔧 Epochs: 1 → 3 (manageable scaling)")
        print("  📚 Text length: 32 → 256 tokens (8x longer)")
        print("  🎯 Expected F1: ~20% → ~45-55% (good progress toward 67-71%)")
        print("")

        print("🏁 LAUNCHING CORRECTED TRAINING...")
        print("=" * 40)
        print("⏱️ Estimated time: 4-6 hours (CPU training)")
        print("💾 Model will be saved to experiments/rebel_final_corrected/")
        print("")

        # Launch training
        train_main(config)

        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("📁 Check experiments/rebel_final_corrected/ for results")
        print("📊 Run evaluation to compare with paper performance")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

        print("\n🔧 TROUBLESHOOTING:")
        print("  1. Verify conda environment: conda activate mining_analytics")
        print("  2. Check available memory (need ~5GB)")
        print("  3. Verify training data files exist")

        return False

if __name__ == "__main__":
    print("🔧 FINAL CORRECTED APPROACH")
    print("Fixing sample size issue + ensuring proper environment")
    print("=" * 60)

    success = run_final_training()

    if success:
        print("\n✅ SUCCESS: Final corrected training completed!")
        print("🎯 Ready to evaluate against paper performance (F1=67-71%)")
        print("📈 Expected significant improvement over minimal training")
    else:
        print("\n❌ FAILED: Check error messages and troubleshooting steps")

    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
🏗️ SPERT SETUP AND TRAINING SCRIPT
Sets up SpERT environment, updates configurations, and runs all 8 SpERT experiments

This script:
1. Creates SpERT virtual environment
2. Updates configuration file paths
3. Launches all SpERT training experiments systematically
"""

import os
import sys
import subprocess
import configparser
from pathlib import Path
import time

class SpERTExperimentRunner:
    """Manages SpERT experiment setup and execution"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.spert_dir = self.base_dir / "models" / "spert"
        self.data_dir = self.base_dir / "models" / "data"
        self.configs_dir = self.spert_dir / "configs"

        # Experiment mapping
        self.experiments = {
            # Direct Fine-tuning (FG)
            "FG-0": {"config": "maintie_g_0_train.conf", "eval_config": "maintie_g_0_eval.conf"},
            "FG-1": {"config": "maintie_g_1_train.conf", "eval_config": "maintie_g_1_eval.conf"},
            "FG-2": {"config": "maintie_g_2_train.conf", "eval_config": "maintie_g_2_eval.conf"},
            "FG-3": {"config": "maintie_g_3_train.conf", "eval_config": "maintie_g_3_eval.conf"},

            # Sequential Fine-tuning (CG+FG) - requires base models
            "CG+FG-0": {"config": "maintie_gs_0_train.conf", "eval_config": "maintie_gs_0_eval.conf", "base": "maintie_s_0_train.conf"},
            "CG+FG-1": {"config": "maintie_gs_1_train.conf", "eval_config": "maintie_gs_1_eval.conf", "base": "maintie_s_1_train.conf"},
            "CG+FG-2": {"config": "maintie_gs_2_train.conf", "eval_config": "maintie_gs_2_eval.conf", "base": "maintie_s_2_train.conf"},
            "CG+FG-3": {"config": "maintie_gs_3_train.conf", "eval_config": "maintie_gs_3_eval.conf", "base": "maintie_s_3_train.conf"},
        }

    def setup_environment(self):
        """Set up SpERT Python environment"""
        print("🏗️ Setting up SpERT environment...")

        # Check if we're already in the mining_analytics environment
        current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
        if current_env == 'mining_analytics':
            print("✅ Already in mining_analytics environment")
            return True
        else:
            print("⚠️  Not in mining_analytics environment")
            print("Please run: conda activate mining_analytics")
            return False

    def update_config_paths(self):
        """Update all SpERT configuration files with correct local paths"""
        print("📝 Updating SpERT configuration file paths...")

        configs_to_update = []

        # Collect all config files
        for config_file in self.configs_dir.glob("*.conf"):
            configs_to_update.append(config_file)

        for config_path in configs_to_update:
            print(f"   Updating {config_path.name}...")

            # Read the configuration
            config = configparser.ConfigParser()
            config.read(config_path)

            # Update paths in each section
            for section in config.sections():
                # Determine dataset level from filename
                if "_0" in config_path.name:
                    level = "0"
                elif "_1" in config_path.name:
                    level = "1"
                elif "_2" in config_path.name:
                    level = "2"
                elif "_3" in config_path.name:
                    level = "3"
                else:
                    continue

                # Determine corpus type (g=gold, s=silver, gs=gold+silver)
                if "_gs_" in config_path.name:
                    train_corpus = "g"  # Training data is gold
                    types_corpus = "s"  # Types from silver corpus
                elif "_g_" in config_path.name:
                    train_corpus = "g"  # Gold corpus
                    types_corpus = "g"  # Gold types
                elif "_s_" in config_path.name:
                    train_corpus = "s"  # Silver corpus
                    types_corpus = "s"  # Silver types
                else:
                    continue

                # Update paths
                base_data_path = str(self.data_dir / f"{train_corpus}-{level}")
                types_data_path = str(self.data_dir / f"{types_corpus}-{level}")

                config[section]['train_path'] = f"{base_data_path}/maintie_train.json"
                config[section]['valid_path'] = f"{base_data_path}/maintie_dev.json"
                config[section]['types_path'] = f"{types_data_path}/maintie_types.json"

                # Update model paths for sequential fine-tuning
                if "_gs_" in config_path.name:
                    # These need to point to the trained silver corpus models
                    base_model_dir = f"data/save/maintie_s_{level}_train"
                    config[section]['model_path'] = base_model_dir
                    config[section]['tokenizer_path'] = base_model_dir
                else:
                    # Direct fine-tuning uses BERT base
                    config[section]['model_path'] = "bert-base-cased"
                    config[section]['tokenizer_path'] = "bert-base-cased"

                # Ensure output directories
                config[section]['log_path'] = "data/log/"
                config[section]['save_path'] = "data/save/"

            # Write updated configuration
            with open(config_path, 'w') as f:
                config.write(f)

        print("✅ All configuration files updated!")

    def create_directories(self):
        """Create necessary directories for SpERT"""
        print("📁 Creating SpERT directories...")

        spert_data_dir = self.spert_dir / "data"
        log_dir = spert_data_dir / "log"
        save_dir = spert_data_dir / "save"

        log_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("✅ Directories created!")

    def run_experiment(self, experiment_name, config_file, background=True):
        """Run a single SpERT experiment"""
        print(f"🚀 Launching SpERT {experiment_name}...")

        cmd = [
            "python", "spert.py", "train",
            "--config", f"configs/{config_file}"
        ]

        if background:
            # Run in background
            process = subprocess.Popen(
                cmd,
                cwd=self.spert_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"   Started {experiment_name} in background (PID: {process.pid})")
            return process
        else:
            # Run in foreground
            result = subprocess.run(
                cmd,
                cwd=self.spert_dir,
                capture_output=True,
                text=True
            )
            return result

    def run_base_model_training(self):
        """Train base models on silver corpus for sequential fine-tuning"""
        print("\n🏗️ PHASE 1: Training Base Models on Silver Corpus")
        print("=" * 60)

        base_experiments = [
            ("Silver-0", "maintie_s_0_train.conf"),
            ("Silver-1", "maintie_s_1_train.conf"),
            ("Silver-2", "maintie_s_2_train.conf"),
            ("Silver-3", "maintie_s_3_train.conf"),
        ]

        processes = []
        for exp_name, config_file in base_experiments:
            process = self.run_experiment(exp_name, config_file, background=True)
            processes.append((exp_name, process))
            time.sleep(2)  # Stagger starts

        return processes

    def run_direct_finetuning(self):
        """Run direct fine-tuning experiments (FG)"""
        print("\n🎯 PHASE 2: Direct Fine-tuning Experiments (FG)")
        print("=" * 60)

        fg_experiments = [
            ("FG-0", "maintie_g_0_train.conf"),
            ("FG-1", "maintie_g_1_train.conf"),
            ("FG-2", "maintie_g_2_train.conf"),
            ("FG-3", "maintie_g_3_train.conf"),
        ]

        processes = []
        for exp_name, config_file in fg_experiments:
            process = self.run_experiment(exp_name, config_file, background=True)
            processes.append((exp_name, process))
            time.sleep(2)  # Stagger starts

        return processes

    def run_sequential_finetuning(self):
        """Run sequential fine-tuning experiments (CG+FG) - requires base models"""
        print("\n🔄 PHASE 3: Sequential Fine-tuning Experiments (CG+FG)")
        print("=" * 60)
        print("⚠️  Note: These require completed base models from Phase 1")

        cgfg_experiments = [
            ("CG+FG-0", "maintie_gs_0_train.conf"),
            ("CG+FG-1", "maintie_gs_1_train.conf"),
            ("CG+FG-2", "maintie_gs_2_train.conf"),
            ("CG+FG-3", "maintie_gs_3_train.conf"),
        ]

        processes = []
        for exp_name, config_file in cgfg_experiments:
            # Check if base model exists
            level = exp_name.split("-")[1]
            base_model_dir = self.spert_dir / "data" / "save" / f"maintie_s_{level}_train"

            if base_model_dir.exists():
                process = self.run_experiment(exp_name, config_file, background=True)
                processes.append((exp_name, process))
                time.sleep(2)
            else:
                print(f"   ⏳ {exp_name} waiting for base model: {base_model_dir}")
                processes.append((exp_name, None))

        return processes

    def monitor_training(self, processes):
        """Monitor training processes"""
        print(f"\n📊 Monitoring {len(processes)} training processes...")

        for exp_name, process in processes:
            if process and process.poll() is None:
                print(f"   ✅ {exp_name} running (PID: {process.pid})")
            elif process and process.poll() is not None:
                print(f"   ❌ {exp_name} finished/failed (exit code: {process.poll()})")
            else:
                print(f"   ⏳ {exp_name} not started")

    def run_all_experiments(self):
        """Run all SpERT experiments systematically"""
        print("🏆 SPERT COMPLETE TRAINING PIPELINE")
        print("=" * 60)

        # Setup
        if not self.setup_environment():
            return

        self.update_config_paths()
        self.create_directories()

        # Phase 1: Base models (required for sequential fine-tuning)
        base_processes = self.run_base_model_training()

        # Phase 2: Direct fine-tuning (can run in parallel with Phase 1)
        fg_processes = self.run_direct_finetuning()

        # Phase 3: Sequential fine-tuning (depends on Phase 1)
        print("\n⏳ Waiting 30 seconds before starting sequential fine-tuning...")
        time.sleep(30)
        cgfg_processes = self.run_sequential_finetuning()

        # Monitor all processes
        all_processes = base_processes + fg_processes + cgfg_processes
        self.monitor_training(all_processes)

        print("\n✅ All SpERT experiments launched!")
        print("📈 Monitor progress in data/log/ directory")

        return all_processes

def main():
    """Main function"""
    runner = SpERTExperimentRunner()
    runner.run_all_experiments()

if __name__ == "__main__":
    main()

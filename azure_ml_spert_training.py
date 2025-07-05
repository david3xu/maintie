# azure_ml_spert_training.py
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Azure ML SpERT training entry point"""
    
    # Setup directories
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    
    # Fix PyTorch Lightning bug
    exec(open('fix_pytorch_lightning.py').read())
    
    # Create datasets
    print("Creating datasets...")
    subprocess.run([sys.executable, "models/create_datasets.py"], check=True)
    
    # Generate Azure ML configs
    print("Generating Azure ML configurations...")
    exec(open('create_azure_configs.py').read())
    
    # Run SpERT training
    print("Starting SpERT training...")
    subprocess.run([
        sys.executable, "models/spert/spert.py", "train",
        "--config", "models/spert/configs/azure-ml/maintie_g_1_train_azureml.conf"
    ], check=True)
    
    print("SpERT training completed successfully")

if __name__ == "__main__":
    main()

# azure_ml_rebel_training.py  
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Azure ML REBEL training entry point"""
    
    # Setup directories
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    
    # Download REBEL base model
    exec(open('download_rebel_model.py').read())
    
    # Update REBEL configurations
    exec(open('update_rebel_configs.py').read())
    
    # Create CPU-optimized configs
    exec(open('optimize_for_cpu.py').read())
    
    # Change to REBEL directory
    os.chdir("models/rebel")
    
    # Run REBEL training with CPU-optimized config
    print("Starting REBEL training...")
    subprocess.run([
        sys.executable, "src/train.py",
        "model=rebel_model",
        "data=maintie_g_1_data_azureml", 
        "train=maintie_g_1_train_cpu"
    ], check=True)
    
    print("REBEL training completed successfully")

if __name__ == "__main__":
    main()

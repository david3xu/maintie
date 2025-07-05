# setup_azure_environment.py
import os
import subprocess
import sys

def setup_azure_ml_environment():
    """Setup Azure ML environment for MaintIE"""
    
    print("Setting up Azure ML environment for MaintIE...")
    
    # Install CPU-optimized PyTorch
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', 
        'torch', 'torchvision', 'torchaudio', 
        '--index-url', 'https://download.pytorch.org/whl/cpu'
    ])
    
    # Install other dependencies
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        '-r', 'requirements_azure_ml.txt'
    ])
    
    # Download spaCy model
    subprocess.check_call([
        sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
    ])
    
    print("Azure ML environment setup complete")

if __name__ == "__main__":
    setup_azure_ml_environment()

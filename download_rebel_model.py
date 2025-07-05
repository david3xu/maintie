# download_rebel_model.py
import os
import subprocess
from pathlib import Path

def download_rebel_base_model():
    """Download REBEL base model for Azure ML environment"""
    
    model_dir = Path("models/rebel/model/Rebel-large")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading REBEL base model...")
    
    # Use git lfs to download from Hugging Face
    if not (model_dir / "pytorch_model.bin").exists():
        subprocess.run([
            "git", "clone", 
            "https://huggingface.co/Babelscape/rebel-large",
            str(model_dir)
        ], check=True)
        
        print(f"REBEL base model downloaded to: {model_dir}")
    else:
        print("REBEL base model already exists")

if __name__ == "__main__":
    download_rebel_base_model()

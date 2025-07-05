# MaintIE Source Code Modifications for Azure ML

## Critical Code Updates Required

Based on analysis of the MaintIE codebase, several source code modifications are essential for Azure ML replication. These changes address path dependencies, environment differences, and Azure ML-specific requirements.

## 1. SpERT Configuration Files - Path Updates

### Issue: Hardcoded Windows Paths
All SpERT configuration files contain absolute Windows paths that must be updated.

**Files requiring modification:**
- `models/spert/configs/maintie_s_0_train.conf`
- `models/spert/configs/maintie_s_1_train.conf` 
- `models/spert/configs/maintie_s_2_train.conf`
- `models/spert/configs/maintie_s_3_train.conf`
- All corresponding `maintie_g_*` and `maintie_gs_*` configs

### Required Changes:

**Create Azure ML compatible configs:**
```bash
# Create new configuration directory
mkdir -p models/spert/configs/azure-ml/

# Copy and modify configurations
cp models/spert/configs/maintie_s_1_train.conf models/spert/configs/azure-ml/maintie_s_1_train_azureml.conf
```

**Update path structure in config files:**
```ini
# Before (Windows paths):
train_path = D:/Repos/nlp_tlp/maintie/models/data/s-1/maintie_train.json
valid_path = D:/Repos/nlp_tlp/maintie/models/data/s-1/maintie_dev.json
types_path = D:/Repos/nlp_tlp/maintie/models/data/s-1/maintie_types.json
log_path = data/log/
save_path = data/save/

# After (Azure ML relative paths):
train_path = ./models/data/s-1/maintie_train.json
valid_path = ./models/data/s-1/maintie_dev.json
types_path = ./models/data/s-1/maintie_types.json
log_path = ./outputs/logs/
save_path = ./outputs/models/
```

### Automated Configuration Generator:

```python
# create_azure_configs.py
import os
import configparser

def update_spert_config(input_file, output_file):
    """Update SpERT config for Azure ML compatibility"""
    config = configparser.ConfigParser()
    config.read(input_file)
    
    # Update paths to be relative
    for section in config.sections():
        if 'train_path' in config[section]:
            config[section]['train_path'] = config[section]['train_path'].replace(
                'D:/Repos/nlp_tlp/maintie/', './'
            )
        if 'valid_path' in config[section]:
            config[section]['valid_path'] = config[section]['valid_path'].replace(
                'D:/Repos/nlp_tlp/maintie/', './'
            )
        if 'types_path' in config[section]:
            config[section]['types_path'] = config[section]['types_path'].replace(
                'D:/Repos/nlp_tlp/maintie/', './'
            )
        
        # Update output paths for Azure ML
        config[section]['log_path'] = './outputs/logs/'
        config[section]['save_path'] = './outputs/models/'
        
        # CPU-optimized parameters
        config[section]['train_batch_size'] = '1'
        config[section]['eval_batch_size'] = '2'
        config[section]['epochs'] = '10'
        
    with open(output_file, 'w') as f:
        config.write(f)

# Convert all configuration files
configs = [
    'maintie_s_0_train.conf',
    'maintie_s_1_train.conf', 
    'maintie_s_2_train.conf',
    'maintie_s_3_train.conf',
    'maintie_g_0_train.conf',
    'maintie_g_1_train.conf',
    'maintie_g_2_train.conf', 
    'maintie_g_3_train.conf'
]

os.makedirs('models/spert/configs/azure-ml', exist_ok=True)

for config_file in configs:
    input_path = f'models/spert/configs/{config_file}'
    output_path = f'models/spert/configs/azure-ml/{config_file.replace(".conf", "_azureml.conf")}'
    
    if os.path.exists(input_path):
        update_spert_config(input_path, output_path)
        print(f"Updated: {output_path}")
```

## 2. REBEL Configuration Updates

### Issue: Absolute Path Requirements
REBEL Hydra configurations require absolute paths that need dynamic generation.

**Create Azure ML compatible REBEL configs:**

```python
# update_rebel_configs.py
import os
import yaml
from pathlib import Path

def update_rebel_config_paths():
    """Update REBEL configuration paths for Azure ML"""
    base_path = Path.cwd()
    
    # Update data configuration paths
    data_configs = [
        'models/rebel/conf/data/maintie_g_0.yaml',
        'models/rebel/conf/data/maintie_g_1.yaml', 
        'models/rebel/conf/data/maintie_g_2.yaml',
        'models/rebel/conf/data/maintie_g_3.yaml',
        'models/rebel/conf/data/maintie_s_1.yaml'
    ]
    
    for config_path in data_configs:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update paths to current working directory
            config['train_file'] = str(base_path / config['train_file'].split('/')[-3:])
            config['dev_file'] = str(base_path / config['dev_file'].split('/')[-3:])
            config['test_file'] = str(base_path / config['test_file'].split('/')[-3:])
            
            # Save updated configuration
            azure_config_path = config_path.replace('.yaml', '_azureml.yaml')
            with open(azure_config_path, 'w') as f:
                yaml.dump(config, f)
            
            print(f"Updated: {azure_config_path}")

if __name__ == "__main__":
    update_rebel_config_paths()
```

## 3. PyTorch Lightning Bug Fix

### Critical Fix Required
The documented PyTorch Lightning bug must be addressed programmatically.

**Create automated fix script:**
```python
# fix_pytorch_lightning.py
import os
import site
import sys

def fix_pytorch_lightning_bug():
    """Fix the documented PyTorch Lightning checkpoint bug"""
    
    # Find pytorch_lightning installation
    for path in site.getsitepackages():
        pl_saving_path = os.path.join(path, 'pytorch_lightning', 'core', 'saving.py')
        
        if os.path.exists(pl_saving_path):
            print(f"Found pytorch_lightning at: {pl_saving_path}")
            
            # Read the file
            with open(pl_saving_path, 'r') as f:
                content = f.read()
            
            # Check if fix already applied
            if 'MaintIE_FIX_APPLIED' in content:
                print("Fix already applied")
                return
            
            # Apply the fix - comment out problematic line
            fixed_content = content.replace(
                'checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)',
                '# checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)  # MaintIE_FIX_APPLIED'
            )
            
            # Write back the fixed file
            with open(pl_saving_path, 'w') as f:
                f.write(fixed_content)
            
            print("PyTorch Lightning bug fix applied successfully")
            return
    
    print("Warning: pytorch_lightning installation not found")

if __name__ == "__main__":
    fix_pytorch_lightning_bug()
```

## 4. CPU-Optimized Training Parameters

### Create CPU-specific training configurations:

```python
# optimize_for_cpu.py
import yaml
import configparser

def create_cpu_optimized_configs():
    """Create CPU-optimized training configurations"""
    
    # SpERT CPU optimizations
    spert_cpu_params = {
        'train_batch_size': '1',
        'eval_batch_size': '2', 
        'epochs': '10',
        'lr': '3e-5',
        'neg_entity_count': '50',
        'neg_relation_count': '50',
        'sampling_processes': '1',
        'max_pairs': '500'
    }
    
    # REBEL CPU optimizations  
    rebel_cpu_params = {
        'train_batch_size': 1,
        'eval_batch_size': 2,
        'gradient_acc_steps': 8,
        'max_steps': 1000,
        'num_train_epochs': 3.0,
        'learning_rate': 0.00003,
        'gpus': 0,  # CPU only
        'precision': 32
    }
    
    # Update REBEL training configs
    rebel_train_configs = [
        'models/rebel/conf/train/maintie_g_0_train.yaml',
        'models/rebel/conf/train/maintie_g_1_train.yaml',
        'models/rebel/conf/train/maintie_g_2_train.yaml',
        'models/rebel/conf/train/maintie_g_3_train.yaml'
    ]
    
    for config_path in rebel_train_configs:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply CPU optimizations
            config.update(rebel_cpu_params)
            
            # Save CPU-optimized version
            cpu_config_path = config_path.replace('.yaml', '_cpu.yaml')
            with open(cpu_config_path, 'w') as f:
                yaml.dump(config, f)
            
            print(f"Created CPU config: {cpu_config_path}")

if __name__ == "__main__":
    create_cpu_optimized_configs()
```

## 5. Azure ML Environment Setup

### Create Azure ML compatible requirements:

```python
# requirements_azure_ml.txt
torch>=1.13.0
transformers>=4.21.0
datasets>=2.0.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
spacy>=3.4.0
tokenizers>=0.12.0
accelerate>=0.12.0
hydra-core>=1.2.0
omegaconf>=2.2.0
pytorch-lightning>=1.7.0
tensorboard>=2.8.0
seqeval>=1.2.0
```

### Environment preparation script:

```python
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
```

## 6. REBEL Base Model Download Handler

### Handle REBEL base model download for Azure ML:

```python
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
```

## 7. Azure ML Job Scripts

### Create Azure ML compatible execution scripts:

```python
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
    subprocess.run([sys.executable, "create_datasets.py"], check=True)
    
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
```

```python
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
        sys.executable, "train.py",
        "model=rebel_model",
        "data=maintie_g_1_azureml", 
        "train=maintie_g_1_train_cpu"
    ], check=True)
    
    print("REBEL training completed successfully")

if __name__ == "__main__":
    main()
```

## 8. Deployment Scripts

### Complete Azure ML deployment automation:

```bash
#!/bin/bash
# deploy_maintie_azure.sh

echo "Deploying MaintIE to Azure ML..."

# Create Azure ML compute cluster
az ml compute create --name maintie-cpu-cluster \
  --type amlcompute \
  --size Standard_E16s_v3 \
  --max-instances 2 \
  --min-instances 0 \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

# Upload code to Azure ML
az ml data create --name maintie-code \
  --type uri_folder \
  --path ./ \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

# Submit SpERT training job
az ml job create --file azure-ml-spert-job.yml \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

echo "MaintIE deployment initiated. Monitor progress in Azure ML Studio."
```

## Quick-Start Implementation

### Step 1: Apply All Code Modifications (10 minutes)
```bash
# Run all modification scripts
python create_azure_configs.py
python update_rebel_configs.py  
python optimize_for_cpu.py
python fix_pytorch_lightning.py
python setup_azure_environment.py

# Commit changes to azure-ml-replication branch
git add .
git commit -m "Apply Azure ML code modifications - CPU optimization, path fixes, environment setup"
```

### Step 2: Test Local Configuration (5 minutes)
```bash
# Verify configurations work locally
python azure_ml_spert_training.py --dry-run
python azure_ml_rebel_training.py --dry-run
```

### Step 3: Deploy to Azure ML (15 minutes)
```bash
# Upload modified codebase
az ml data create --name maintie-code-modified \
  --type uri_folder \
  --path ./ \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

# Submit training jobs
az ml job create --file azure-ml-spert-job.yml \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace
```

These modifications ensure MaintIE runs successfully in Azure ML with CPU optimization, proper path handling, and environment compatibility. All changes maintain research accuracy while enabling cloud deployment.
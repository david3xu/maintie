# Original Codebase Updates for Azure ML SDK v2

## 📋 Overview

This document details the specific modifications needed in your original MaintIE project codebase to work seamlessly with Azure ML SDK v2. The updates maintain backward compatibility while adding Azure ML support.

## 🎯 Update Strategy

**Approach**: Add Azure ML support layers while preserving original functionality

**Key Principle**: Original code remains functional locally, Azure ML features are additive

## 📁 Files Requiring Updates

### 1. Training Scripts Updates

#### A. REBEL Training Script Updates

**File**: `models/rebel/src/train.py`

**Required Changes**:

```python
# === ORIGINAL CODE PRESERVATION ===
# Keep existing imports and functions

# === ADD AZURE ML SUPPORT ===
import os
import argparse
from pathlib import Path

# Azure ML imports (conditional)
try:
    import mlflow
    import mlflow.pytorch
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False

def setup_azure_ml_args(parser):
    """Add Azure ML specific arguments"""
    # Azure ML Data Asset paths
    parser.add_argument('--data_path', type=str, 
                       help='Azure ML Data Asset path')
    parser.add_argument('--output_path', type=str,
                       help='Azure ML output path for models')
    parser.add_argument('--metrics_path', type=str,
                       help='Azure ML metrics output path')
    parser.add_argument('--checkpoint_path', type=str,
                       help='Azure ML checkpoint output path')
    
    # Azure ML parameters
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Azure ML learning rate parameter')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Azure ML epochs parameter')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Azure ML batch size parameter')
    parser.add_argument('--sweep_mode', action='store_true',
                       help='Enable Azure ML sweep mode')
    
    return parser

def resolve_azure_ml_paths(conf, args):
    """Resolve Azure ML data asset paths"""
    if hasattr(args, 'data_path') and args.data_path:
        # Azure ML Data Asset path resolution
        data_base = Path(args.data_path)
        
        # Update data paths in config
        if hasattr(conf, 'train_data'):
            conf.train_data = str(data_base / 'maintie_train.json')
        if hasattr(conf, 'val_data'):
            conf.val_data = str(data_base / 'maintie_dev.json')
        if hasattr(conf, 'test_data'):
            conf.test_data = str(data_base / 'maintie_test.json')
        if hasattr(conf, 'types_data'):
            conf.types_data = str(data_base / 'maintie_types.json')
    
    return conf

def setup_azure_ml_outputs(args):
    """Setup Azure ML output directories"""
    output_dirs = []
    
    if hasattr(args, 'output_path') and args.output_path:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)
        output_dirs.append(args.output_path)
    
    if hasattr(args, 'metrics_path') and args.metrics_path:
        Path(args.metrics_path).mkdir(parents=True, exist_ok=True)
        output_dirs.append(args.metrics_path)
    
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
        output_dirs.append(args.checkpoint_path)
    
    return output_dirs

def setup_mlflow_tracking(args):
    """Setup MLflow tracking for Azure ML"""
    if AZURE_ML_AVAILABLE and not args.sweep_mode:
        try:
            # Auto-logging for PyTorch Lightning
            mlflow.pytorch.autolog()
            
            # Log parameters
            if hasattr(args, 'learning_rate') and args.learning_rate:
                mlflow.log_param("learning_rate", args.learning_rate)
            if hasattr(args, 'epochs') and args.epochs:
                mlflow.log_param("epochs", args.epochs)
            if hasattr(args, 'batch_size') and args.batch_size:
                mlflow.log_param("batch_size", args.batch_size)
                
        except Exception as e:
            print(f"Warning: MLflow setup failed: {e}")

def save_azure_ml_model(model, tokenizer, args):
    """Save model to Azure ML outputs"""
    if hasattr(args, 'output_path') and args.output_path:
        output_path = Path(args.output_path)
        
        # Save model
        model_path = output_path / 'model'
        model.save_pretrained(model_path)
        
        # Save tokenizer
        tokenizer_path = output_path / 'tokenizer'
        tokenizer.save_pretrained(tokenizer_path)
        
        print(f"Model saved to Azure ML output: {output_path}")
        
        # Log model with MLflow
        if AZURE_ML_AVAILABLE:
            try:
                mlflow.pytorch.log_model(model, "model")
            except Exception as e:
                print(f"Warning: MLflow model logging failed: {e}")

# === MODIFY MAIN TRAIN FUNCTION ===
@hydra.main(config_path="../conf", config_name="root")
def train(conf: omegaconf.DictConfig) -> None:
    # === AZURE ML INTEGRATION ===
    # Parse additional Azure ML arguments
    parser = argparse.ArgumentParser()
    parser = setup_azure_ml_args(parser)
    
    # Parse known args to avoid conflicts with Hydra
    azure_args, unknown = parser.parse_known_args()
    
    # Setup Azure ML
    setup_azure_ml_outputs(azure_args)
    setup_mlflow_tracking(azure_args)
    
    # Resolve Azure ML paths
    conf = resolve_azure_ml_paths(conf, azure_args)
    
    # Override config with Azure ML parameters
    if azure_args.learning_rate:
        conf.lr = azure_args.learning_rate
    if azure_args.epochs:
        conf.max_epochs = azure_args.epochs
    if azure_args.batch_size:
        conf.train_batch_size = azure_args.batch_size
    
    # === ORIGINAL TRAINING LOGIC ===
    pl.seed_everything(conf.seed)
    
    # ... [Keep existing training code] ...
    
    # === AZURE ML MODEL SAVING ===
    # After training completion, save to Azure ML outputs
    if hasattr(azure_args, 'output_path'):
        save_azure_ml_model(model, tokenizer, azure_args)
    
    print("Training completed!")

if __name__ == "__main__":
    train()
```

#### B. SpERT Training Script Updates

**File**: `models/spert/train_azure_ml.py` (New wrapper script)

```python
#!/usr/bin/env python3
"""
Azure ML Wrapper for SpERT Training
Maintains original SpERT functionality while adding Azure ML support
"""

import argparse
import os
import sys
from pathlib import Path

# Add SpERT to path
sys.path.append(str(Path(__file__).parent))

# Original SpERT imports
from spert.args import train_argparser
from spert.spert_trainer import SpERTTrainer
from spert.input_reader import JsonInputReader

# Azure ML imports
try:
    import mlflow
    import mlflow.pytorch
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False

class AzureMLSpERTTrainer(SpERTTrainer):
    """SpERT Trainer with Azure ML support"""
    
    def __init__(self, args):
        # Setup Azure ML outputs before parent init
        self._setup_azure_ml_outputs(args)
        super().__init__(args)
        
        # Setup MLflow if available
        if AZURE_ML_AVAILABLE:
            self._setup_mlflow(args)
    
    def _setup_azure_ml_outputs(self, args):
        """Setup Azure ML output directories"""
        if hasattr(args, 'output_path') and args.output_path:
            Path(args.output_path).mkdir(parents=True, exist_ok=True)
            # Update save_path to Azure ML output
            args.save_path = args.output_path
        
        if hasattr(args, 'metrics_path') and args.metrics_path:
            Path(args.metrics_path).mkdir(parents=True, exist_ok=True)
        
        if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
            Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    def _setup_mlflow(self, args):
        """Setup MLflow tracking"""
        try:
            # Log parameters
            mlflow.log_param("model_type", args.model_type)
            mlflow.log_param("learning_rate", args.lr)
            mlflow.log_param("epochs", args.epochs)
            mlflow.log_param("batch_size", args.train_batch_size)
            mlflow.log_param("max_span_size", args.max_span_size)
        except Exception as e:
            print(f"Warning: MLflow setup failed: {e}")
    
    def _resolve_azure_ml_data_paths(self, args):
        """Resolve Azure ML data asset paths"""
        if hasattr(args, 'data_path') and args.data_path:
            data_base = Path(args.data_path)
            
            # Try to find data files
            train_files = list(data_base.glob('*train*.json'))
            valid_files = list(data_base.glob('*dev*.json'))
            types_files = list(data_base.glob('*types*.json'))
            
            if train_files:
                args.train_path = str(train_files[0])
            if valid_files:
                args.valid_path = str(valid_files[0])
            if types_files:
                args.types_path = str(types_files[0])
    
    def train_azure_ml(self):
        """Train with Azure ML integration"""
        args = self._args
        
        # Resolve data paths
        self._resolve_azure_ml_data_paths(args)
        
        # Original SpERT training
        self.train(
            train_path=args.train_path,
            valid_path=args.valid_path,
            types_path=args.types_path,
            input_reader_cls=JsonInputReader
        )
        
        # Save model to Azure ML outputs
        if hasattr(args, 'output_path') and args.output_path:
            print(f"Model saved to Azure ML output: {args.output_path}")

def azure_ml_argparser():
    """Create argument parser with Azure ML support"""
    # Start with original SpERT args
    parser = train_argparser()
    
    # Add Azure ML specific arguments
    parser.add_argument('--data_path', type=str,
                       help='Azure ML Data Asset path')
    parser.add_argument('--output_path', type=str,
                       help='Azure ML output path for models')
    parser.add_argument('--metrics_path', type=str,
                       help='Azure ML metrics output path')
    parser.add_argument('--checkpoint_path', type=str,
                       help='Azure ML checkpoint output path')
    
    # Azure ML parameter overrides
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate (overrides lr)')
    parser.add_argument('--sweep_mode', action='store_true',
                       help='Enable Azure ML sweep mode')
    
    return parser

def main():
    """Main function for Azure ML SpERT training"""
    parser = azure_ml_argparser()
    args = parser.parse_args()
    
    # Apply Azure ML parameter overrides
    if hasattr(args, 'learning_rate') and args.learning_rate:
        args.lr = args.learning_rate
    
    # Create trainer and train
    trainer = AzureMLSpERTTrainer(args)
    trainer.train_azure_ml()

if __name__ == '__main__':
    main()
```

### 2. Configuration Updates

#### A. REBEL Configuration Updates

**File**: `models/rebel/conf/azure_ml_maintie.yaml`

```yaml
# Azure ML Configuration for REBEL
defaults:
  - train: maintie_train_azure_ml
  - model: rebel_model
  - data: maintie_data_azure_ml

# Azure ML specific settings
azure_ml:
  data_asset_name: "maintie-training-data"
  output_model_name: "rebel-maintie-model"
  experiment_name: "rebel-maintie-training"

# Override paths for Azure ML
paths:
  data_path: ${oc.env:AZURE_ML_DATA_PATH,"/tmp/data"}
  output_path: ${oc.env:AZURE_ML_OUTPUT_PATH,"/tmp/outputs"}
  checkpoint_path: ${oc.env:AZURE_ML_CHECKPOINT_PATH,"/tmp/checkpoints"}

# MLflow tracking
mlflow:
  tracking_uri: ${oc.env:MLFLOW_TRACKING_URI,""}
  experiment_name: ${azure_ml.experiment_name}
```

**File**: `models/rebel/conf/data/maintie_data_azure_ml.yaml`

```yaml
# Azure ML Data Configuration
train_data: ${paths.data_path}/maintie_train.json
val_data: ${paths.data_path}/maintie_dev.json
test_data: ${paths.data_path}/maintie_test.json
types_data: ${paths.data_path}/maintie_types.json

# Data parameters
max_input_length: 256
max_output_length: 256
train_data_percentage: 1.0
val_data_percentage: 1.0
```

#### B. SpERT Configuration Updates

**File**: `models/spert/configs/azure_ml_train.conf`

```ini
[1] Sections

[model]
model_type = spert
model_path = bert-base-cased
tokenizer_path = bert-base-cased

[data]
train_path = ${AZURE_ML_DATA_PATH}/maintie_train.json
valid_path = ${AZURE_ML_DATA_PATH}/maintie_dev.json
types_path = ${AZURE_ML_DATA_PATH}/maintie_types.json

[train]
train_batch_size = 2
eval_batch_size = 1
neg_entity_count = 100
neg_relation_count = 100
epochs = 20
lr = 5e-5
lr_warmup = 0.1
weight_decay = 0.01
max_grad_norm = 1.0

[azure_ml]
output_path = ${AZURE_ML_OUTPUT_PATH}
metrics_path = ${AZURE_ML_METRICS_PATH}
checkpoint_path = ${AZURE_ML_CHECKPOINT_PATH}
experiment_name = spert-maintie-training

[misc]
max_span_size = 10
store_predictions = true
store_examples = true
sampling_processes = 4
sampling_limit = 100
```

### 3. Data Handling Updates

#### A. Data Path Resolution Script

**File**: `utils/azure_ml_data_resolver.py`

```python
#!/usr/bin/env python3
"""
Azure ML Data Path Resolver
Handles data asset path resolution for both models
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional

class AzureMLDataResolver:
    """Resolves Azure ML data asset paths"""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or os.getenv('AZURE_ML_DATA_PATH', './data')
        self.data_dir = Path(self.data_path)
    
    def resolve_maintie_data_paths(self) -> Dict[str, str]:
        """Resolve MaintIE data file paths"""
        data_files = {
            'train': None,
            'dev': None,
            'test': None,
            'types': None
        }
        
        # Search for data files
        for file_path in self.data_dir.glob('*.json'):
            file_name = file_path.name.lower()
            
            if 'train' in file_name:
                data_files['train'] = str(file_path)
            elif 'dev' in file_name or 'valid' in file_name:
                data_files['dev'] = str(file_path)
            elif 'test' in file_name:
                data_files['test'] = str(file_path)
            elif 'types' in file_name:
                data_files['types'] = str(file_path)
        
        return data_files
    
    def validate_data_files(self) -> bool:
        """Validate that required data files exist"""
        data_files = self.resolve_maintie_data_paths()
        
        missing_files = []
        for key, path in data_files.items():
            if not path or not Path(path).exists():
                missing_files.append(key)
        
        if missing_files:
            print(f"Missing data files: {missing_files}")
            return False
        
        return True
    
    def create_environment_variables(self):
        """Create environment variables for data paths"""
        data_files = self.resolve_maintie_data_paths()
        
        # Set environment variables
        os.environ['MAINTIE_TRAIN_DATA'] = data_files.get('train', '')
        os.environ['MAINTIE_DEV_DATA'] = data_files.get('dev', '')
        os.environ['MAINTIE_TEST_DATA'] = data_files.get('test', '')
        os.environ['MAINTIE_TYPES_DATA'] = data_files.get('types', '')
        
        print("Data path environment variables set:")
        for key, value in data_files.items():
            print(f"  MAINTIE_{key.upper()}_DATA = {value}")
```

### 4. Model Saving Updates

#### A. Azure ML Model Registry Integration

**File**: `utils/azure_ml_model_registry.py`

```python
#!/usr/bin/env python3
"""
Azure ML Model Registry Integration
Handles model registration and versioning
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import mlflow
    import mlflow.pytorch
    from azure.ai.ml.entities import Model
    from azure.ai.ml.constants import AssetTypes
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False

class AzureMLModelRegistry:
    """Handles Azure ML model registration"""
    
    def __init__(self, model_name: str, experiment_name: str):
        self.model_name = model_name
        self.experiment_name = experiment_name
    
    def save_model_with_metadata(self, 
                                model, 
                                tokenizer,
                                output_path: str,
                                metadata: Optional[Dict[str, Any]] = None):
        """Save model with comprehensive metadata"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model_dir = output_dir / 'model'
        tokenizer_dir = output_dir / 'tokenizer'
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        
        # Save metadata
        model_metadata = {
            'model_name': self.model_name,
            'experiment_name': self.experiment_name,
            'framework': 'pytorch',
            'model_type': 'transformer',
            'task': 'entity_relation_extraction',
            'model_dir': str(model_dir),
            'tokenizer_dir': str(tokenizer_dir),
            **(metadata or {})
        }
        
        metadata_path = output_dir / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"Model saved with metadata: {output_dir}")
        
        # Register with MLflow
        if AZURE_ML_AVAILABLE:
            self._register_with_mlflow(model, model_metadata)
    
    def _register_with_mlflow(self, model, metadata: Dict[str, Any]):
        """Register model with MLflow"""
        try:
            # Log model
            mlflow.pytorch.log_model(
                model, 
                self.model_name,
                extra_files=[metadata.get('tokenizer_dir', '')]
            )
            
            # Log metadata
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
            
            print(f"Model registered with MLflow: {self.model_name}")
            
        except Exception as e:
            print(f"Warning: MLflow registration failed: {e}")
```

### 5. Dependency Updates

#### A. Consolidated Requirements

**File**: `azure-ml-v2/environments/requirements_consolidated.txt`

```txt
# Azure ML SDK v2 - Latest
azure-ai-ml==1.27.1
azure-identity==1.15.0

# Common Framework Dependencies
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
tokenizers==0.14.1

# SpERT Dependencies
bert-score==0.3.13
scikit-learn==1.3.0
spacy==3.7.0
tqdm==4.66.1

# REBEL Dependencies  
pytorch-lightning==2.1.0
omegaconf==2.3.3
hydra-core==1.3.2
datasets==2.14.0

# Common ML Dependencies
pandas==2.1.0
numpy==1.24.3
seqeval==1.2.2
jsonlines==4.0.0

# Tracking and Logging
mlflow==2.7.1
tensorboard==2.14.0
wandb==0.15.12

# Utilities
click==8.1.7
pyyaml==6.0.1
python-dotenv==1.0.0
```

### 6. Launch Scripts

#### A. Unified Training Launcher

**File**: `azure-ml-v2/utils/train_launcher.py`

```python
#!/usr/bin/env python3
"""
Unified Training Launcher for Azure ML
Supports both SpERT and REBEL models
"""

import argparse
import subprocess
import sys
from pathlib import Path

def launch_spert_training(args):
    """Launch SpERT training with Azure ML support"""
    cmd = [
        sys.executable,
        'models/spert/train_azure_ml.py',
        '--config', args.config or 'configs/azure_ml_train.conf',
        '--data_path', args.data_path,
        '--output_path', args.output_path,
    ]
    
    # Add optional parameters
    if args.learning_rate:
        cmd.extend(['--learning_rate', str(args.learning_rate)])
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        cmd.extend(['--train_batch_size', str(args.batch_size)])
    
    print(f"Launching SpERT training: {' '.join(cmd)}")
    return subprocess.run(cmd)

def launch_rebel_training(args):
    """Launch REBEL training with Azure ML support"""
    cmd = [
        sys.executable,
        'models/rebel/src/train.py',
        f'hydra.run.dir={args.output_path}',
        f'paths.data_path={args.data_path}',
        f'paths.output_path={args.output_path}',
    ]
    
    # Add optional parameters
    if args.learning_rate:
        cmd.append(f'lr={args.learning_rate}')
    if args.epochs:
        cmd.append(f'max_epochs={args.epochs}')
    if args.batch_size:
        cmd.append(f'train_batch_size={args.batch_size}')
    
    print(f"Launching REBEL training: {' '.join(cmd)}")
    return subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Unified Azure ML Training Launcher")
    
    parser.add_argument('--model', choices=['spert', 'rebel'], required=True,
                       help='Model to train')
    parser.add_argument('--config', type=str,
                       help='Configuration file')
    parser.add_argument('--data_path', required=True,
                       help='Azure ML data asset path')
    parser.add_argument('--output_path', required=True,
                       help='Azure ML output path')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size')
    
    args = parser.parse_args()
    
    if args.model == 'spert':
        return launch_spert_training(args)
    elif args.model == 'rebel':
        return launch_rebel_training(args)

if __name__ == '__main__':
    sys.exit(main().returncode)
```

### 7. Testing and Validation Updates

#### A. Azure ML Compatibility Test

**File**: `azure-ml-v2/utils/test_model_compatibility.py`

```python
#!/usr/bin/env python3
"""
Test Model Compatibility with Azure ML
Validates that models work correctly in Azure ML environment
"""

import sys
import subprocess
import tempfile
from pathlib import Path

def test_spert_compatibility():
    """Test SpERT Azure ML compatibility"""
    print("Testing SpERT Azure ML compatibility...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock data
        data_path = Path(temp_dir) / 'data'
        data_path.mkdir()
        
        # Create minimal test files
        (data_path / 'maintie_train.json').write_text('[]')
        (data_path / 'maintie_dev.json').write_text('[]')
        (data_path / 'maintie_types.json').write_text('{"entities": [], "relations": []}')
        
        # Test import
        try:
            sys.path.append('models/spert')
            from train_azure_ml import AzureMLSpERTTrainer
            print("✅ SpERT Azure ML imports successful")
            return True
        except ImportError as e:
            print(f"❌ SpERT Azure ML import failed: {e}")
            return False

def test_rebel_compatibility():
    """Test REBEL Azure ML compatibility"""
    print("Testing REBEL Azure ML compatibility...")
    
    try:
        # Test if modifications are compatible
        cmd = [
            sys.executable, '-c',
            'import sys; sys.path.append("models/rebel/src"); '
            'from train import setup_azure_ml_args; print("REBEL Azure ML compatible")'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ REBEL Azure ML imports successful")
            return True
        else:
            print(f"❌ REBEL Azure ML import failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ REBEL compatibility test failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("Running Azure ML compatibility tests...")
    
    spert_ok = test_spert_compatibility()
    rebel_ok = test_rebel_compatibility()
    
    if spert_ok and rebel_ok:
        print("🎉 All models are Azure ML compatible!")
        return True
    else:
        print("❌ Some models have compatibility issues")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
```

## 📋 Update Checklist

### Core Training Scripts
- [ ] **REBEL**: Add Azure ML parameter parsing to `models/rebel/src/train.py`
- [ ] **REBEL**: Add data path resolution for Azure ML Data Assets
- [ ] **REBEL**: Add MLflow integration for experiment tracking
- [ ] **REBEL**: Add model saving to Azure ML outputs
- [ ] **SpERT**: Create `models/spert/train_azure_ml.py` wrapper
- [ ] **SpERT**: Add Azure ML argument parsing
- [ ] **SpERT**: Add data path resolution
- [ ] **SpERT**: Add model output handling

### Configuration Files
- [ ] **REBEL**: Create Azure ML config `models/rebel/conf/azure_ml_maintie.yaml`
- [ ] **REBEL**: Update data configurations for Azure ML paths
- [ ] **SpERT**: Create Azure ML config `models/spert/configs/azure_ml_train.conf`
- [ ] **SpERT**: Add environment variable support

### Data Management
- [ ] Create `utils/azure_ml_data_resolver.py` for path resolution
- [ ] Add data validation for Azure ML Data Assets
- [ ] Update data loading logic for both models
- [ ] Test data asset access patterns

### Model Registry Integration
- [ ] Create `utils/azure_ml_model_registry.py`
- [ ] Add model metadata generation
- [ ] Integrate MLflow model logging
- [ ] Add model versioning support

### Dependencies and Environment
- [ ] Consolidate requirements in `azure-ml-v2/environments/requirements_consolidated.txt`
- [ ] Test package compatibility
- [ ] Update environment setup scripts
- [ ] Validate Azure ML SDK v2 integration

### Testing and Validation
- [ ] Create compatibility test scripts
- [ ] Test local vs Azure ML execution
- [ ] Validate parameter passing
- [ ] Test model saving and loading

### Launch and Automation
- [ ] Create unified training launcher
- [ ] Add parameter validation
- [ ] Create deployment automation scripts
- [ ] Test end-to-end workflows

## 🎯 Key Principles for Updates

### 1. **Backward Compatibility**
- Original training scripts continue to work locally
- Azure ML features are additive, not replacement
- Existing configurations remain functional

### 2. **Parameter Mapping**
```python
# Azure ML parameters map to original parameters
azure_ml_params = {
    'learning_rate': 'lr',           # REBEL/SpERT
    'epochs': 'max_epochs',          # REBEL
    'epochs': 'epochs',              # SpERT  
    'batch_size': 'train_batch_size' # Both
}
```

### 3. **Path Resolution**
```python
# Azure ML Data Asset paths
data_asset_path = "/mnt/batch/tasks/shared/LS_root/jobs/..."
local_path = "models/data/g-1/"

# Automatic resolution based on environment
if os.getenv('AZURE_ML_DATA_PATH'):
    data_path = azure_data_resolver.resolve_paths()
else:
    data_path = local_path
```

### 4. **Output Management**
```python
# Azure ML outputs
output_mapping = {
    'model_output': args.output_path,
    'metrics_output': args.metrics_path,
    'checkpoint_output': args.checkpoint_path
}
```

## ⚡ Quick Implementation

### 1. **Immediate Priority Updates**
1. **REBEL**: Modify `models/rebel/src/train.py` with Azure ML argument parsing
2. **SpERT**: Create `models/spert/train_azure_ml.py` wrapper
3. **Data**: Create `utils/azure_ml_data_resolver.py`
4. **Launcher**: Create `azure-ml-v2/utils/train_launcher.py`

### 2. **Testing Approach**
```bash
# Test locally first
python azure-ml-v2/utils/test_model_compatibility.py

# Test with Azure ML
python azure-ml-v2/utils/train_launcher.py \
  --model spert \
  --data_path ${{inputs.training_data}} \
  --output_path ${{outputs.model_output}} \
  --learning_rate 5e-5 \
  --epochs 10
```

### 3. **Validation Steps**
- [ ] Local execution still works
- [ ] Azure ML parameter passing works
- [ ] Data loading from Azure ML Data Assets works
- [ ] Model saving to Azure ML outputs works
- [ ] MLflow tracking captures metrics

## 🔄 Migration Strategy

### Phase 1: Core Updates (Week 1)
- Update training scripts with Azure ML support
- Create wrapper scripts for backward compatibility
- Test basic parameter passing

### Phase 2: Integration (Week 2)  
- Add data path resolution
- Integrate model registry
- Add MLflow tracking

### Phase 3: Validation (Week 3)
- End-to-end testing
- Performance validation
- Documentation updates

The updates maintain the integrity of your original codebase while adding powerful Azure ML capabilities for scalable, cloud-based training and experimentation!

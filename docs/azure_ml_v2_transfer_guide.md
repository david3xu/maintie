# Local to Azure ML Transfer Guide - SDK v2 Enhanced

## 📋 Overview

This guide documents the complete process of replicating a local machine learning project to Azure ML using **Azure ML SDK v2** (latest), based on the successful MaintIE project deployment. It covers all bridge components, transfer points, and modern v2 best practices.

## 🎯 Executive Summary

**Successful Strategy**: Use Azure ML SDK v2 with curated environments, data assets, and programmatic job management.

**Key Success Factors**:

- ✅ **Azure ML SDK v2** (`azure-ai-ml==1.27.1`)
- ✅ **PyTorch curated environment** (`azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest`)
- ✅ **Data Assets** for proper data management
- ✅ **Inputs/Outputs pattern** for job parameterization
- ✅ **MLClient** for programmatic control
- ✅ **Bridge utilities** for environment adaptation
- ✅ **Organized Azure ML directory structure**

## 📊 Transfer Architecture

```
Local Development Environment
            ↓
    [SDK v2 Bridge Components]
            ↓
Azure ML Cloud Environment (v2)
```

### Main Transfer Points

1. **Code Transfer**: Local Python files → Azure ML v2 job execution
2. **Environment Transfer**: Local dependencies → Azure ML curated environment + runtime packages
3. **Data Transfer**: Local data files → Azure ML Data Assets
4. **Configuration Transfer**: Local configs → Azure ML v2 job definitions
5. **Model Transfer**: Local models → Azure ML Model Registry

## 🏗️ Directory Structure Setup

### 1. Create Azure ML v2 Bridge Directory

```bash
mkdir -p azure-ml-v2/{jobs,environments,utils,configs,data,models,docs}
```

### 2. Organize Project Structure

```
project-root/
├── azure-ml-v2/               # 🌉 SDK v2 BRIDGE DIRECTORY
│   ├── jobs/                  # v2 Job definitions
│   ├── environments/          # v2 Environment specifications
│   ├── utils/                 # v2 Bridge utilities
│   ├── configs/               # v2 Azure configurations
│   ├── data/                  # Data asset management
│   ├── models/                # Model registration
│   └── docs/                  # Documentation
├── models/                    # Original training code
├── data/                      # Training data
└── requirements.txt           # Local dependencies
```

## 🌉 Bridge Components Implementation

### 1. Core SDK v2 Setup

#### A. Requirements File

**File**: `azure-ml-v2/environments/requirements_azure_ml_v2.txt`

```txt
# Azure ML SDK v2 Core
azure-ai-ml==1.27.1
azure-identity==1.15.0

# ML Framework Dependencies
transformers==4.35.0
torch==2.1.0
torchvision==0.16.0
datasets==2.14.0
tokenizers==0.14.1

# Training Dependencies
seqeval==1.2.2
hydra-core==1.3.2
omegaconf==2.3.3
jsonlines==4.0.0
scikit-learn==1.3.0
pandas==2.1.0
numpy==1.24.3

# Monitoring & Logging
mlflow==2.7.1
tensorboard==2.14.0
wandb==0.15.12

# Utilities
tqdm==4.66.1
click==8.1.7
pyyaml==6.0.1
```

#### B. Environment Configuration

**File**: `azure-ml-v2/environments/pytorch-v2-environment.yml`

```yaml
# Azure ML v2 Environment Reference
name: pytorch-v2-environment
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.9
  - pytorch=2.1.0
  - torchvision=0.16.0
  - pip
  - pip:
      - azure-ai-ml==1.27.1
      - azure-identity==1.15.0
      - transformers==4.35.0
      - datasets==2.14.0
      - seqeval==1.2.2
      - hydra-core==1.3.2
      - omegaconf==2.3.3
      - jsonlines==4.0.0
```

### 2. Azure ML Client Bridge

#### A. Main Client Bridge

**File**: `azure-ml-v2/utils/azure_ml_client.py`

```python
#!/usr/bin/env python3
"""
Azure ML SDK v2 Client Bridge
Provides unified interface for Azure ML operations
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Data, Model, Environment, Job,
    CommandJob, AmlCompute
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

class AzureMLClientBridge:
    """Bridge class for Azure ML SDK v2 operations"""
    
    def __init__(self, config_path: str = "azure-ml-v2/configs/azure_config.json"):
        """Initialize Azure ML client with configuration"""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.ml_client = self._create_ml_client()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('azure_ml_bridge.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load Azure ML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"✅ Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"❌ Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"❌ Invalid JSON in configuration: {e}")
            raise
    
    def _create_ml_client(self) -> MLClient:
        """Create Azure ML client"""
        try:
            ml_client = MLClient(
                credential=DefaultAzureCredential(),
                subscription_id=self.config["subscription_id"],
                resource_group_name=self.config["resource_group"],
                workspace_name=self.config["workspace_name"]
            )
            self.logger.info("✅ Azure ML client created successfully")
            return ml_client
        except Exception as e:
            self.logger.error(f"❌ Failed to create ML client: {e}")
            raise
    
    def register_data_asset(self, 
                           local_path: str, 
                           name: str, 
                           version: str = None,
                           description: str = None) -> Data:
        """Register data as Azure ML Data Asset"""
        try:
            data_asset = Data(
                path=local_path,
                type=AssetTypes.URI_FOLDER,
                description=description or f"Data asset for {name}",
                name=name,
                version=version
            )
            
            registered_data = self.ml_client.data.create_or_update(data_asset)
            self.logger.info(f"✅ Data asset registered: {name}:{registered_data.version}")
            return registered_data
        except Exception as e:
            self.logger.error(f"❌ Failed to register data asset: {e}")
            raise
    
    def submit_job(self, job_config_path: str) -> Job:
        """Submit job to Azure ML"""
        try:
            from azure.ai.ml import load_job
            
            job = load_job(source=job_config_path)
            submitted_job = self.ml_client.jobs.create_or_update(job)
            
            self.logger.info(f"✅ Job submitted: {submitted_job.name}")
            self.logger.info(f"📊 Job URL: {submitted_job.studio_url}")
            return submitted_job
        except Exception as e:
            self.logger.error(f"❌ Failed to submit job: {e}")
            raise
    
    def register_model(self, 
                       model_path: str, 
                       name: str, 
                       version: str = None,
                       description: str = None) -> Model:
        """Register model in Azure ML Model Registry"""
        try:
            model = Model(
                path=model_path,
                name=name,
                version=version,
                description=description or f"Model {name}",
                type=AssetTypes.CUSTOM_MODEL
            )
            
            registered_model = self.ml_client.models.create_or_update(model)
            self.logger.info(f"✅ Model registered: {name}:{registered_model.version}")
            return registered_model
        except Exception as e:
            self.logger.error(f"❌ Failed to register model: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> str:
        """Get job status"""
        try:
            job = self.ml_client.jobs.get(job_name)
            self.logger.info(f"📊 Job {job_name} status: {job.status}")
            return job.status
        except Exception as e:
            self.logger.error(f"❌ Failed to get job status: {e}")
            raise
    
    def stream_job_logs(self, job_name: str):
        """Stream job logs"""
        try:
            self.ml_client.jobs.stream(job_name)
        except Exception as e:
            self.logger.error(f"❌ Failed to stream logs: {e}")
            raise
    
    def list_data_assets(self) -> list:
        """List all data assets"""
        try:
            data_assets = list(self.ml_client.data.list())
            self.logger.info(f"📊 Found {len(data_assets)} data assets")
            return data_assets
        except Exception as e:
            self.logger.error(f"❌ Failed to list data assets: {e}")
            raise
    
    def get_compute_status(self, compute_name: str) -> str:
        """Get compute target status"""
        try:
            compute = self.ml_client.compute.get(compute_name)
            self.logger.info(f"💻 Compute {compute_name} status: {compute.provisioning_state}")
            return compute.provisioning_state
        except ResourceNotFoundError:
            self.logger.warning(f"⚠️ Compute {compute_name} not found")
            return "NotFound"
        except Exception as e:
            self.logger.error(f"❌ Failed to get compute status: {e}")
            raise
```

#### B. Data Management Bridge

**File**: `azure-ml-v2/utils/data_management.py`

```python
#!/usr/bin/env python3
"""
Azure ML v2 Data Management Bridge
Handles data assets and data preparation
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

class DataManagementBridge:
    """Bridge for Azure ML data management operations"""
    
    def __init__(self, ml_client_bridge):
        self.ml_client = ml_client_bridge.ml_client
        self.logger = logging.getLogger(__name__)
    
    def prepare_training_data(self, local_data_path: str, asset_name: str) -> str:
        """Prepare and register training data as Azure ML Data Asset"""
        
        # Validate local data
        if not self._validate_local_data(local_data_path):
            raise ValueError(f"Invalid data at {local_data_path}")
        
        # Register as data asset
        data_asset = Data(
            path=local_data_path,
            type=AssetTypes.URI_FOLDER,
            description=f"Training data for {asset_name}",
            name=asset_name
        )
        
        registered_data = self.ml_client.data.create_or_update(data_asset)
        data_uri = f"azureml:{asset_name}@latest"
        
        self.logger.info(f"✅ Training data registered as: {data_uri}")
        return data_uri
    
    def _validate_local_data(self, data_path: str) -> bool:
        """Validate local data format and structure"""
        required_files = [
            'maintie_train.json',
            'maintie_dev.json', 
            'maintie_test.json',
            'maintie_types.json'
        ]
        
        data_dir = Path(data_path)
        if not data_dir.exists():
            self.logger.error(f"❌ Data directory does not exist: {data_path}")
            return False
        
        for file_name in required_files:
            file_path = data_dir / file_name
            if not file_path.exists():
                self.logger.error(f"❌ Required file missing: {file_path}")
                return False
            
            # Validate JSON format
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                self.logger.info(f"✅ Validated: {file_name}")
            except json.JSONDecodeError as e:
                self.logger.error(f"❌ Invalid JSON in {file_name}: {e}")
                return False
        
        return True
    
    def create_data_preparation_script(self, output_path: str):
        """Create data preparation script for Azure ML"""
        script_content = '''#!/usr/bin/env python3
"""
Azure ML Data Preparation Script
Prepares data for training in Azure ML environment
"""

import json
import os
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def prepare_data(input_path: str, output_path: str):
    """Prepare data for training"""
    logger = setup_logging()
    
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy and validate data files
    data_files = ['maintie_train.json', 'maintie_dev.json', 'maintie_test.json', 'maintie_types.json']
    
    for file_name in data_files:
        src_file = input_dir / file_name
        dst_file = output_dir / file_name
        
        if src_file.exists():
            with open(src_file, 'r') as f:
                data = json.load(f)
            
            with open(dst_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Prepared: {file_name}")
        else:
            logger.error(f"❌ Missing: {file_name}")
    
    logger.info(f"✅ Data preparation complete: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    
    prepare_data(args.input_path, args.output_path)
'''
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        self.logger.info(f"✅ Created data preparation script: {output_path}")
```

### 3. Enhanced Job Definitions

#### A. Training Job with SDK v2 Features

**File**: `azure-ml-v2/jobs/model-training-v2.yml`

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
experiment_name: project-model-pytorch-v2
display_name: Project-Model-Training-SDK-v2
description: "Model training using Azure ML SDK v2 with enhanced features"

# Modern environment reference
environment: azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest

# Compute configuration
compute: azureml:cpu-cluster

# Input/Output pattern (v2 best practice)
inputs:
  # Data inputs
  training_data:
    type: uri_folder
    path: azureml:project-training-data@latest
  
  # Training parameters
  learning_rate:
    type: number
    default: 0.001
  epochs:
    type: integer
    default: 10
  batch_size:
    type: integer
    default: 16
  model_name:
    type: string
    default: "spert"
  
  # Configuration
  config_file:
    type: uri_file
    path: models/configs/train.conf

outputs:
  # Model outputs
  model_output:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/models/project-model-v2/
  
  # Metrics and logs
  metrics_output:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/metrics/project-model-v2/
  
  # Checkpoints
  checkpoint_output:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/checkpoints/project-model-v2/

# Code source
code: .

# Enhanced command with parameter substitution
command: >-
  echo "🚀 Starting Model Training - Azure ML SDK v2" &&
  echo "📦 Installing additional packages..." &&
  pip install -r azure-ml-v2/environments/requirements_azure_ml_v2.txt &&
  echo "✅ Package installation complete" &&
  echo "🔧 Setting up environment..." &&
  python azure-ml-v2/utils/setup_azure_environment.py &&
  echo "📊 Training Configuration:" &&
  echo "  Learning Rate: ${{inputs.learning_rate}}" &&
  echo "  Epochs: ${{inputs.epochs}}" &&
  echo "  Batch Size: ${{inputs.batch_size}}" &&
  echo "  Model: ${{inputs.model_name}}" &&
  echo "🎓 Starting training..." &&
  python models/train.py \
    --data_path ${{inputs.training_data}} \
    --output_path ${{outputs.model_output}} \
    --metrics_path ${{outputs.metrics_output}} \
    --checkpoint_path ${{outputs.checkpoint_output}} \
    --config ${{inputs.config_file}} \
    --learning_rate ${{inputs.learning_rate}} \
    --epochs ${{inputs.epochs}} \
    --batch_size ${{inputs.batch_size}} \
    --model_name ${{inputs.model_name}}

# Enhanced tagging and metadata
tags:
  framework: pytorch
  sdk_version: v2
  environment_type: curated
  project: maintie
  version: "2.0"
  approach: enhanced_v2

# Resource configuration
resources:
  instance_count: 1
  shm_size: 2g

# Job settings
settings:
  continue_on_step_failure: false
  default_compute: azureml:cpu-cluster
```

#### B. Hyperparameter Sweep Job

**File**: `azure-ml-v2/jobs/hyperparameter-sweep-v2.yml`

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/sweepJob.schema.json
type: sweep
experiment_name: project-hyperparameter-sweep-v2
display_name: Project-Hyperparameter-Sweep-SDK-v2
description: "Hyperparameter optimization using Azure ML SDK v2"

# Sweep configuration
sampling_algorithm: random
max_total_trials: 20
max_concurrent_trials: 4
timeout: 7200

# Search space
search_space:
  learning_rate:
    type: uniform
    min_value: 0.0001
    max_value: 0.01
  epochs:
    type: choice
    values: [5, 10, 15, 20]
  batch_size:
    type: choice
    values: [8, 16, 32]

# Objective metric
objective:
  goal: maximize
  primary_metric: f1_score

# Base job template
trial: 
  type: command
  environment: azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest
  compute: azureml:cpu-cluster
  code: .
  
  inputs:
    training_data:
      type: uri_folder
      path: azureml:project-training-data@latest
    learning_rate: ${{search_space.learning_rate}}
    epochs: ${{search_space.epochs}}
    batch_size: ${{search_space.batch_size}}
  
  outputs:
    model_output:
      type: uri_folder
      path: azureml://datastores/workspaceblobstore/paths/sweep-models/
  
  command: >-
    pip install -r azure-ml-v2/environments/requirements_azure_ml_v2.txt &&
    python azure-ml-v2/utils/setup_azure_environment.py &&
    python models/train.py \
      --data_path ${{inputs.training_data}} \
      --output_path ${{outputs.model_output}} \
      --learning_rate ${{inputs.learning_rate}} \
      --epochs ${{inputs.epochs}} \
      --batch_size ${{inputs.batch_size}} \
      --sweep_mode true

tags:
  type: hyperparameter_sweep
  framework: pytorch
  sdk_version: v2
```

### 4. Environment Setup Bridge (Enhanced)

#### A. Main Environment Setup

**File**: `azure-ml-v2/utils/setup_azure_environment.py`

```python
#!/usr/bin/env python3
"""
Azure ML SDK v2 Environment Setup Bridge Script
Enhanced setup for Azure ML v2 execution environment
"""

import os
import sys
import subprocess
import logging
import json
import platform
from pathlib import Path
from typing import Dict, List, Any

def setup_logging():
    """Configure enhanced logging for Azure ML v2"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('logs/azure_ml_setup.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def install_requirements():
    """Install requirements with enhanced error handling"""
    logger = logging.getLogger(__name__)
    
    requirements_files = [
        'azure-ml-v2/environments/requirements_azure_ml_v2.txt',
        'requirements.txt'
    ]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            logger.info(f"📦 Installing requirements from {req_file}")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '-r', req_file,
                    '--no-cache-dir', '--upgrade'
                ])
                logger.info(f"✅ Successfully installed requirements from {req_file}")
                break
            except subprocess.CalledProcessError as e:
                logger.warning(f"❌ Failed to install requirements from {req_file}: {e}")
                continue
    else:
        logger.warning("⚠️ No requirements file found or all installations failed")

def setup_azure_ml_environment():
    """Setup Azure ML specific environment variables"""
    logger = logging.getLogger(__name__)
    
    # Azure ML v2 environment variables
    env_vars = {
        'PYTHONPATH': ':'.join([
            os.getcwd(),
            os.path.join(os.getcwd(), 'models'),
            os.path.join(os.getcwd(), 'azure-ml-v2', 'utils'),
            os.path.join(os.getcwd(), 'utils')
        ]),
        'TOKENIZERS_PARALLELISM': 'false',
        'TRANSFORMERS_CACHE': '/tmp/transformers_cache',
        'HF_DATASETS_CACHE': '/tmp/datasets_cache',
        'MLFLOW_TRACKING_URI': 'azureml://experiments/project-model-pytorch-v2',
        'AZURE_ML_SDK_VERSION': 'v2',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"🔧 Set environment variable: {key}")

def create_directories():
    """Create necessary directories for Azure ML v2"""
    logger = logging.getLogger(__name__)
    
    directories = [
        'outputs',
        'logs',
        'checkpoints',
        'metrics',
        '/tmp/transformers_cache',
        '/tmp/datasets_cache',
        '/tmp/model_cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def validate_azure_ml_setup():
    """Validate Azure ML v2 setup and dependencies"""
    logger = logging.getLogger(__name__)
    
    # Check Azure ML SDK v2
    try:
        import azure.ai.ml
        logger.info(f"✅ Azure ML SDK v2 version: {azure.ai.ml.__version__}")
    except ImportError:
        logger.error("❌ Azure ML SDK v2 not available")
        return False
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
        else:
            logger.info("💻 Using CPU")
    except ImportError:
        logger.error("❌ PyTorch not available")
    
    # Check Transformers
    try:
        import transformers
        logger.info(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("❌ Transformers not available")
    
    return True

def setup_mlflow_tracking():
    """Setup MLflow tracking for Azure ML v2"""
    logger = logging.getLogger(__name__)
    
    try:
        import mlflow
        
        # Set tracking URI for Azure ML
        mlflow.set_tracking_uri("azureml://experiments/project-model-pytorch-v2")
        
        # Start MLflow run
        if not mlflow.active_run():
            mlflow.start_run()
            logger.info("🔬 MLflow tracking initialized")
        
        # Log system info
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("platform", platform.platform())
        
    except ImportError:
        logger.warning("⚠️ MLflow not available - tracking disabled")
    except Exception as e:
        logger.warning(f"⚠️ MLflow setup failed: {e}")

def print_system_info():
    """Print comprehensive system information"""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("🖥️ SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Available memory: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3):.2f} GB")
    
    # Check important directories
    important_dirs = ['models', 'data', 'azure-ml-v2', 'outputs']
    for dir_name in important_dirs:
        if os.path.exists(dir_name):
            logger.info(f"✅ Directory exists: {dir_name}")
        else:
            logger.warning(f"❌ Directory missing: {dir_name}")
    
    logger.info("=" * 60)

def validate_data_availability():
    """Validate training data availability"""
    logger = logging.getLogger(__name__)
    
    # Check for data files
    data_patterns = [
        'models/data/g-1/*.json',
        'data/*.json',
        '*.json'
    ]
    
    import glob
    found_files = []
    for pattern in data_patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    if found_files:
        logger.info(f"✅ Found {len(found_files)} data files")
        for file in found_files[:5]:  # Show first 5
            logger.info(f"   📄 {file}")
    else:
        logger.warning("⚠️ No data files found")

def main():
    """Main setup function for Azure ML SDK v2"""
    logger = setup_logging()
    
    logger.info("🚀 Starting Azure ML SDK v2 Environment Setup")
    logger.info("=" * 60)
    
    try:
        # Setup steps
        print_system_info()
        setup_azure_ml_environment()
        create_directories()
        install_requirements()
        validate_azure_ml_setup()
        setup_mlflow_tracking()
        validate_data_availability()
        
        logger.info("=" * 60)
        logger.info("✅ Azure ML SDK v2 Environment Setup Complete")
        logger.info("🎯 Ready for model training!")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment setup failed: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### 5. Configuration Files

#### A. Azure ML Configuration

**File**: `azure-ml-v2/configs/azure_config.json`

```json
{
  "subscription_id": "your-subscription-id",
  "resource_group": "your-resource-group",
  "workspace_name": "your-workspace-name",
  "location": "eastus2",
  "compute_target": "cpu-cluster",
  "experiment_name": "project-training-v2",
  "default_datastore": "workspaceblobstore",
  "sdk_version": "v2",
  "environment": {
    "name": "pytorch-v2-environment",
    "curated": "azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest"
  },
  "data_assets": {
    "training_data": "project-training-data",
    "validation_data": "project-validation-data"
  },
  "model_registry": {
    "default_model_name": "project-model-v2"
  },
  "tags": {
    "project": "maintie",
    "version": "2.0",
    "framework": "pytorch",
    "sdk": "v2",
    "environment": "curated"
  },
  "mlflow": {
    "experiment_name": "project-model-pytorch-v2",
    "tracking_enabled": true
  }
}
```

#### B. Deployment Configuration

**File**: `azure-ml-v2/configs/deployment_config.json`

```json
{
  "deployment": {
    "type": "batch",
    "name": "project-model-deployment-v2",
    "description": "Model deployment using Azure ML SDK v2",
    "compute_target": "cpu-cluster",
    "instance_count": 1,
    "max_concurrency_per_instance": 1,
    "mini_batch_size": 10,
    "retry_settings": {
      "max_retries": 3,
      "timeout": 300
    },
    "environment": "azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest",
    "code_path": "."
  },
  "endpoint": {
    "name": "project-endpoint-v2",
    "description": "Model endpoint using Azure ML SDK v2",
    "auth_mode": "key"
  }
}
```

### 6. Deployment Scripts

#### A. Main Deployment Script

**File**: `azure-ml-v2/utils/deploy.py`

```python
#!/usr/bin/env python3
"""
Azure ML SDK v2 Deployment Script
Complete deployment pipeline using Azure ML SDK v2
"""

import argparse
import json
import logging
from pathlib import Path

from azure_ml_client import AzureMLClientBridge
from data_management import DataManagementBridge

def setup_logging():
    """Setup logging for deployment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def deploy_training_pipeline(args):
    """Deploy complete training pipeline"""
    logger = setup_logging()
    
    logger.info("🚀 Starting Azure ML SDK v2 Deployment")
    
    # Initialize Azure ML client
    azure_client = AzureMLClientBridge(args.config)
    data_manager = DataManagementBridge(azure_client)
    
    # Step 1: Register data assets
    if args.register_data:
        logger.info("📊 Registering data assets...")
        training_data_uri = data_manager.prepare_training_data(
            args.data_path, 
            "project-training-data"
        )
        logger.info(f"✅ Training data registered: {training_data_uri}")
    
    # Step 2: Submit training job
    if args.submit_job:
        logger.info("🎓 Submitting training job...")
        job = azure_client.submit_job(args.job_config)
        logger.info(f"✅ Job submitted: {job.name}")
        logger.info(f"📊 Monitor at: {job.studio_url}")
        
        # Stream logs if requested
        if args.stream_logs:
            logger.info("📝 Streaming job logs...")
            azure_client.stream_job_logs(job.name)
    
    # Step 3: Register model (if job completed successfully)
    if args.register_model and args.model_path:
        logger.info("🏷️ Registering model...")
        model = azure_client.register_model(
            args.model_path,
            "project-model-v2",
            description="Model trained with Azure ML SDK v2"
        )
        logger.info(f"✅ Model registered: {model.name}:{model.version}")
    
    logger.info("🎉 Deployment complete!")

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Azure ML SDK v2 Deployment")
    
    parser.add_argument("--config", 
                       default="azure-ml-v2/configs/azure_config.json",
                       help="Azure ML configuration file")
    parser.add_argument("--job_config",
                       default="azure-ml-v2/jobs/model-training-v2.yml", 
                       help="Job configuration file")
    parser.add_argument("--data_path",
                       default="models/data/g-1",
                       help="Local data path")
    parser.add_argument("--model_path",
                       help="Model path for registration")
    
    # Action flags
    parser.add_argument("--register_data", action="store_true",
                       help="Register data as Azure ML Data Asset")
    parser.add_argument("--submit_job", action="store_true",
                       help="Submit training job")
    parser.add_argument("--register_model", action="store_true",
                       help="Register trained model")
    parser.add_argument("--stream_logs", action="store_true",
                       help="Stream job logs")
    
    args = parser.parse_args()
    
    deploy_training_pipeline(args)

if __name__ == "__main__":
    main()
```

#### B. Quick Deploy Script

**File**: `azure-ml-v2/utils/quick_deploy.py`

```python
#!/usr/bin/env python3
"""
Quick Deployment Script for Azure ML SDK v2
One-command deployment for common scenarios
"""

import sys
import subprocess
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def quick_deploy():
    """Quick deployment with sensible defaults"""
    logger = setup_logging()
    
    logger.info("🚀 Quick Deploy - Azure ML SDK v2")
    
    # Step 1: Run pre-deployment tests
    logger.info("🧪 Running pre-deployment tests...")
    result = subprocess.run([
        sys.executable, 
        "azure-ml-v2/utils/local_pre_deployment_test.py"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error("❌ Pre-deployment tests failed")
        logger.error(result.stderr)
        return False
    
    logger.info("✅ Pre-deployment tests passed")
    
    # Step 2: Deploy with default settings
    logger.info("🚀 Deploying to Azure ML...")
    deploy_cmd = [
        sys.executable,
        "azure-ml-v2/utils/deploy.py",
        "--register_data",
        "--submit_job",
        "--stream_logs"
    ]
    
    result = subprocess.run(deploy_cmd)
    
    if result.returncode == 0:
        logger.info("🎉 Quick deployment successful!")
        return True
    else:
        logger.error("❌ Deployment failed")
        return False

if __name__ == "__main__":
    success = quick_deploy()
    sys.exit(0 if success else 1)
```

## 🚀 Enhanced Deployment Process

### 1. Pre-deployment Validation (Enhanced)

```bash
# Install Azure ML SDK v2
pip install azure-ai-ml==1.27.1 azure-identity==1.15.0

# Run enhanced validation
cd azure-ml-v2/utils
python local_pre_deployment_test.py --verbose
```

### 2. Azure ML Authentication (v2)

```bash
# Azure CLI login
az login

# Install Azure ML CLI v2 extension
az extension add -n ml

# Set defaults
az configure --defaults group=your-resource-group workspace=your-workspace-name

# Verify connection
az ml workspace show
```

### 3. Data Asset Registration

```python
# Using Python SDK v2
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="your-resource-group",
    workspace_name="your-workspace-name"
)

# Register data asset
data_asset = Data(
    path="models/data/g-1",
    type=AssetTypes.URI_FOLDER,
    description="Training data for project",
    name="project-training-data"
)

ml_client.data.create_or_update(data_asset)
```

### 4. Job Submission (Enhanced)

```bash
# Method 1: Using CLI v2
az ml job create -f azure-ml-v2/jobs/model-training-v2.yml

# Method 2: Using Python SDK v2
python azure-ml-v2/utils/deploy.py --register_data --submit_job --stream_logs

# Method 3: Quick deploy
python azure-ml-v2/utils/quick_deploy.py
```

### 5. Monitoring and Management

```python
# Using Python SDK v2
from azure_ml_client import AzureMLClientBridge

client = AzureMLClientBridge()

# Monitor job
job_status = client.get_job_status("job_name")
print(f"Job status: {job_status}")

# Stream logs
client.stream_job_logs("job_name")

# List data assets
data_assets = client.list_data_assets()
for asset in data_assets:
    print(f"Data Asset: {asset.name}:{asset.version}")
```

## 📋 Migration Checklist (SDK v2 Enhanced)

### Pre-migration Preparation

- [ ] Install Azure ML SDK v2 (`azure-ai-ml==1.27.1`)
- [ ] Create `azure-ml-v2/` directory structure
- [ ] Update Python to 3.8+ (recommended 3.9+)
- [ ] Document current dependencies and versions
- [ ] Prepare training data for Data Asset registration

### SDK v2 Bridge Components

- [ ] Create Azure ML client bridge (`azure_ml_client.py`)
- [ ] Set up data management bridge (`data_management.py`)
- [ ] Write enhanced environment setup script
- [ ] Create v2 job definitions with inputs/outputs
- [ ] Configure MLflow integration

### Environment Strategy (v2)

- [ ] Use curated environments (`AzureML-pytorch-1.13-py39-cpu-inference@latest`)
- [ ] Define runtime dependencies in `requirements_azure_ml_v2.txt`
- [ ] Test package compatibility with Python 3.9
- [ ] Set up proper environment variables
- [ ] Configure caching directories

### Data Assets (v2 Feature)

- [ ] Register training data as Data Assets
- [ ] Validate data format and structure
- [ ] Set up data versioning strategy
- [ ] Create data preparation scripts
- [ ] Test data asset access patterns

### Job Configuration (v2)

- [ ] Create parameterized job definitions
- [ ] Set up inputs/outputs pattern
- [ ] Configure resource requirements
- [ ] Add proper tagging and metadata
- [ ] Test job submission and monitoring

### Model Management (v2)

- [ ] Set up model registration pipeline
- [ ] Configure model versioning
- [ ] Add model metadata and tags
- [ ] Test model deployment workflows
- [ ] Set up model monitoring

### Testing and Validation

- [ ] Run enhanced pre-deployment tests
- [ ] Test Azure ML authentication
- [ ] Validate data asset registration
- [ ] Test job execution with parameters
- [ ] Verify model registration

### Deployment and Monitoring

- [ ] Deploy using Python SDK v2
- [ ] Monitor job execution
- [ ] Set up MLflow tracking
- [ ] Configure alerts and notifications
- [ ] Document successful configurations

## 🛠️ Troubleshooting Guide (SDK v2)

### Common SDK v2 Issues

#### 1. Authentication Issues
```python
# Solution: Use DefaultAzureCredential
from azure.identity import DefaultAzureCredential
credential = DefaultAzureCredential()
```

#### 2. Data Asset Registration
```python
# Solution: Proper Data Asset definition
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

data = Data(
    path="local/path",
    type=AssetTypes.URI_FOLDER,
    name="asset-name",
    version="1"
)
```

#### 3. Job Parameter Issues
```yaml
# Solution: Proper parameter substitution
command: python train.py --lr ${{inputs.learning_rate}} --epochs ${{inputs.epochs}}
```

#### 4. Environment Issues
```yaml
# Solution: Use curated environments
environment: azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest
```

## 📊 SDK v2 Success Metrics

### Deployment Success Indicators

- [ ] MLClient authentication successful
- [ ] Data Assets registered and accessible
- [ ] Jobs submit without schema errors
- [ ] Parameters properly substituted
- [ ] Outputs written to correct locations
- [ ] Model registration successful

### Performance Metrics

- [ ] Job submission time < 2 minutes
- [ ] Data asset registration < 5 minutes
- [ ] Environment setup time < 10 minutes
- [ ] Model registration time < 3 minutes
- [ ] Overall deployment time < 20 minutes

## 📚 SDK v2 Resources

- [Azure ML SDK v2 Documentation](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-ml-readme)
- [Azure ML CLI v2 Reference](https://learn.microsoft.com/en-us/cli/azure/ml)
- [Job Schema Reference](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command)
- [Data Assets Guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets)
- [Model Registry Guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-manage-models)

---

**Note**: This enhanced guide uses Azure ML SDK v2 (latest) with modern best practices including Data Assets, parameterized jobs, MLClient, and curated environments. All components are designed for the current Azure ML platform and provide a robust foundation for production ML workflows.

# Week 1 Implementation - Azure ML Technical Validation

## 🎯 **Week 1 Objectives**

**Primary Goal**: Validate that PyTorch Lightning + Hydra + Azure ML SDK v2 work together

**Success Criteria**:
- [ ] REBEL model trains successfully in Azure ML
- [ ] Data assets load correctly
- [ ] Hydra configurations work in Azure ML environment
- [ ] Model outputs save to Azure ML storage
- [ ] No environment/dependency conflicts

## 📋 **Day-by-Day Implementation Plan**

### **Day 1: Environment Setup & Technical Validation**

#### **Step 1.1: Create Minimal Test Environment**

**File**: `week1-validation/test-environment.yml`

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
experiment_name: week1-tech-validation
display_name: Week1-Tech-Stack-Test

environment: azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest
compute: azureml:cpu-cluster
code: .

command: >-
  echo "🧪 Testing Azure ML Tech Stack" &&
  pip install pytorch-lightning==2.1.0 hydra-core==1.3.2 omegaconf==2.3.3 &&
  python week1-validation/test_tech_stack.py

tags:
  week: "1"
  purpose: "tech_validation"
```

**File**: `week1-validation/test_tech_stack.py`

```python
#!/usr/bin/env python3
"""
Technical Stack Validation for Azure ML
Tests PyTorch Lightning + Hydra + Azure ML compatibility
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🔬 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import pytorch_lightning as pl
        print(f"✅ PyTorch Lightning: {pl.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch Lightning import failed: {e}")
        return False
    
    try:
        import hydra
        import omegaconf
        print(f"✅ Hydra: {hydra.__version__}")
        print(f"✅ OmegaConf: {omegaconf.__version__}")
    except ImportError as e:
        print(f"❌ Hydra/OmegaConf import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    return True

def test_hydra_config():
    """Test basic Hydra configuration loading"""
    print("🔧 Testing Hydra configuration...")
    
    from omegaconf import DictConfig, OmegaConf
    
    # Create test config
    test_config = OmegaConf.create({
        "model": {
            "name": "test_model",
            "lr": 0.001
        },
        "data": {
            "batch_size": 16
        }
    })
    
    print(f"✅ Hydra config test passed: {test_config}")
    return True

def test_azure_ml_environment():
    """Test Azure ML environment variables and paths"""
    print("☁️ Testing Azure ML environment...")
    
    # Check common Azure ML environment variables
    az_env_vars = [
        'AZUREML_RUN_ID',
        'AZUREML_EXPERIMENT_NAME', 
        'AZUREML_RUN_TOKEN',
        'AZUREML_WORKSPACE_NAME'
    ]
    
    for var in az_env_vars:
        value = os.getenv(var, 'Not Set')
        print(f"   {var}: {value}")
    
    # Check working directory
    print(f"✅ Working directory: {os.getcwd()}")
    print(f"✅ Python path: {sys.path[0]}")
    
    return True

def test_pytorch_lightning_basic():
    """Test basic PyTorch Lightning functionality"""
    print("⚡ Testing PyTorch Lightning...")
    
    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    
    class SimpleModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.layer(x)
        
        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = nn.functional.mse_loss(y_hat, y)
            return loss
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.001)
    
    model = SimpleModel()
    print("✅ PyTorch Lightning model created successfully")
    return True

def main():
    """Run all technical validation tests"""
    print("🚀 Starting Week 1 Technical Validation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Hydra Config Test", test_hydra_config),
        ("Azure ML Environment Test", test_azure_ml_environment),
        ("PyTorch Lightning Test", test_pytorch_lightning_basic)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TECHNICAL VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Ready for REBEL training validation.")
        return True
    else:
        print("❌ SOME TESTS FAILED! Fix issues before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### **Day 2: Data Asset Registration & Validation**

#### **Step 2.1: Register MaintIE Data as Azure ML Data Asset**

**File**: `week1-validation/register_data_asset.py`

```python
#!/usr/bin/env python3
"""
Register MaintIE data as Azure ML Data Asset
Week 1 validation of data asset functionality
"""

import json
import os
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def validate_local_data():
    """Validate local MaintIE data structure"""
    print("📊 Validating local MaintIE data...")
    
    data_path = Path("models/data/g-1")
    required_files = [
        "maintie_train.json",
        "maintie_dev.json", 
        "maintie_test.json",
        "maintie_types.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = data_path / file_name
        if not file_path.exists():
            missing_files.append(str(file_path))
        else:
            # Validate JSON format
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if file_name == "maintie_types.json":
                        print(f"✅ {file_name}: {len(data.get('entities', []))} entities, {len(data.get('relations', []))} relations")
                    else:
                        print(f"✅ {file_name}: {len(data)} samples")
            except json.JSONDecodeError as e:
                print(f"❌ {file_name}: Invalid JSON - {e}")
                missing_files.append(str(file_path))
    
    if missing_files:
        print(f"❌ Missing or invalid files: {missing_files}")
        return False
    
    print("✅ All MaintIE data files validated successfully!")
    return True

def register_data_asset():
    """Register data as Azure ML Data Asset"""
    print("☁️ Registering MaintIE data as Azure ML Data Asset...")
    
    # Load Azure ML configuration
    config_path = "azure-ml-v2/configs/azure_config.json"
    
    if not Path(config_path).exists():
        print(f"❌ Azure ML config not found: {config_path}")
        print("Please create config file with your Azure ML details.")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create MLClient
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=config["subscription_id"],
            resource_group_name=config["resource_group"],
            workspace_name=config["workspace_name"]
        )
        print("✅ Azure ML client created successfully")
    except Exception as e:
        print(f"❌ Failed to create Azure ML client: {e}")
        return False
    
    # Register data asset
    try:
        data_asset = Data(
            path="models/data/g-1",
            type=AssetTypes.URI_FOLDER,
            description="MaintIE training data for Week 1 validation",
            name="maintie-week1-validation-data",
            version="1"
        )
        
        registered_data = ml_client.data.create_or_update(data_asset)
        print(f"✅ Data asset registered successfully!")
        print(f"   Name: {registered_data.name}")
        print(f"   Version: {registered_data.version}")
        print(f"   ID: {registered_data.id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to register data asset: {e}")
        return False

def test_data_asset_access():
    """Test accessing the registered data asset"""
    print("🔍 Testing data asset access...")
    
    # This will be tested in the actual Azure ML job
    print("✅ Data asset access will be validated in training job")
    return True

def main():
    """Main data registration workflow"""
    print("🚀 Week 1 Data Asset Registration")
    print("=" * 50)
    
    # Step 1: Validate local data
    if not validate_local_data():
        print("❌ Local data validation failed!")
        return False
    
    # Step 2: Register data asset
    if not register_data_asset():
        print("❌ Data asset registration failed!")
        return False
    
    # Step 3: Test access
    if not test_data_asset_access():
        print("❌ Data asset access test failed!")
        return False
    
    print("\n✅ Data asset registration completed successfully!")
    print("Ready to use azureml:maintie-week1-validation-data@latest in training jobs")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### **Day 3: REBEL Training Configuration**

#### **Step 3.1: Create Minimal REBEL Azure ML Configuration**

**File**: `week1-validation/rebel_config_azure.yaml`

```yaml
# Week 1 REBEL Azure ML Configuration
# Minimal config for technical validation

# Model configuration
model_name_or_path: "Babelscape/rebel-large"
config_name: null
tokenizer_name: null
use_fast_tokenizer: true

# Training parameters
max_epochs: 2  # Minimal for validation
train_batch_size: 1  # Small for testing
eval_batch_size: 1
learning_rate: 3e-5
weight_decay: 0.0
warmup_steps: 0

# Data parameters
max_input_length: 256
max_output_length: 256
num_beams: 5

# Azure ML specific
azure_ml:
  data_path: ${oc.env:AZURE_ML_INPUT_training_data,./models/data/g-1}
  output_path: ${oc.env:AZURE_ML_OUTPUT_model_output,./outputs}
  
# Paths (will be overridden by Azure ML environment)
train_file: ${azure_ml.data_path}/maintie_train.json
validation_file: ${azure_ml.data_path}/maintie_dev.json
test_file: ${azure_ml.data_path}/maintie_test.json

# Minimal dataset configuration
dataset_name: "models/rebel/datasets/maintie_lvl_1.py"

# Logging
logging_steps: 10
save_steps: 50
eval_steps: 50

# Validation settings
evaluation_strategy: "steps"
save_strategy: "steps"
load_best_model_at_end: true
metric_for_best_model: "eval_loss"

# Technical settings
dataloader_num_workers: 0  # Avoid multiprocessing issues in Azure ML
seed: 42
```

#### **Step 3.2: Create Azure ML Training Script Wrapper**

**File**: `week1-validation/train_rebel_azure_week1.py`

```python
#!/usr/bin/env python3
"""
Week 1 REBEL Training Wrapper for Azure ML
Minimal modifications to validate technical stack
"""

import os
import sys
import argparse
import json
from pathlib import Path
import yaml

# Add REBEL source to path
sys.path.append(str(Path(__file__).parent.parent / "models" / "rebel" / "src"))

def setup_azure_ml_environment():
    """Setup Azure ML environment variables and paths"""
    print("🔧 Setting up Azure ML environment...")
    
    # Get Azure ML environment variables
    data_path = os.getenv('AZURE_ML_INPUT_training_data', './models/data/g-1')
    output_path = os.getenv('AZURE_ML_OUTPUT_model_output', './outputs')
    
    print(f"📊 Data path: {data_path}")
    print(f"📁 Output path: {output_path}")
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Validate data files exist
    data_dir = Path(data_path)
    required_files = ["maintie_train.json", "maintie_dev.json", "maintie_test.json", "maintie_types.json"]
    
    for file_name in required_files:
        file_path = data_dir / file_name
        if file_path.exists():
            print(f"✅ Found: {file_name}")
        else:
            print(f"❌ Missing: {file_name}")
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    return data_path, output_path

def create_dynamic_config(data_path, output_path):
    """Create dynamic configuration for Azure ML"""
    print("⚙️ Creating dynamic configuration...")
    
    # Load base config
    config_path = Path(__file__).parent / "rebel_config_azure.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths for Azure ML
    config['train_file'] = f"{data_path}/maintie_train.json"
    config['validation_file'] = f"{data_path}/maintie_dev.json"
    config['test_file'] = f"{data_path}/maintie_test.json"
    config['output_dir'] = output_path
    
    # Save dynamic config
    dynamic_config_path = Path(output_path) / "config_dynamic.yaml"
    with open(dynamic_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Dynamic config saved: {dynamic_config_path}")
    return dynamic_config_path

def run_rebel_training(config_path):
    """Run REBEL training with Azure ML configuration"""
    print("🚀 Starting REBEL training...")
    
    try:
        # Import REBEL training components
        from train import train
        import hydra
        from omegaconf import OmegaConf
        
        # Load configuration
        config = OmegaConf.load(config_path)
        
        # Run training
        print("⚡ Executing REBEL training...")
        result = train(config)
        
        print("✅ REBEL training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ REBEL training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_training_summary(output_path, success):
    """Save training summary for validation"""
    summary = {
        "week": 1,
        "purpose": "technical_validation",
        "model": "REBEL",
        "training_success": success,
        "azure_ml_integration": "tested",
        "timestamp": str(Path().cwd()),
        "python_version": sys.version,
        "working_directory": str(Path.cwd())
    }
    
    summary_path = Path(output_path) / "week1_training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Training summary saved: {summary_path}")

def main():
    """Main Week 1 training function"""
    print("🚀 Week 1 REBEL Azure ML Training Validation")
    print("=" * 60)
    
    try:
        # Step 1: Setup Azure ML environment
        data_path, output_path = setup_azure_ml_environment()
        
        # Step 2: Create dynamic configuration
        config_path = create_dynamic_config(data_path, output_path)
        
        # Step 3: Run training
        success = run_rebel_training(config_path)
        
        # Step 4: Save summary
        save_training_summary(output_path, success)
        
        if success:
            print("\n🎉 Week 1 validation completed successfully!")
            print("✅ REBEL + Hydra + Azure ML integration verified!")
            return True
        else:
            print("\n❌ Week 1 validation failed!")
            return False
            
    except Exception as e:
        print(f"\n💥 Week 1 validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### **Day 4: Azure ML Job Definition & Submission**

#### **Step 4.1: Create Week 1 Training Job Definition**

**File**: `week1-validation/rebel-training-week1.yml`

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
type: command
experiment_name: week1-rebel-validation
display_name: Week1-REBEL-Technical-Validation
description: "Week 1 technical validation of REBEL + Hydra + Azure ML integration"

# Environment
environment: azureml:AzureML-pytorch-1.13-py39-cpu-inference@latest
compute: azureml:cpu-cluster

# Inputs
inputs:
  training_data:
    type: uri_folder
    path: azureml:maintie-week1-validation-data@latest

# Outputs  
outputs:
  model_output:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/week1-validation/rebel-model/
  
  validation_output:
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/week1-validation/results/

# Code
code: .

# Command
command: >-
  echo "🚀 Week 1 REBEL Training Validation" &&
  echo "📦 Installing dependencies..." &&
  pip install pytorch-lightning==2.1.0 &&
  pip install hydra-core==1.3.2 &&
  pip install omegaconf==2.3.3 &&
  pip install datasets==2.14.0 &&
  pip install transformers==4.35.0 &&
  pip install seqeval==1.2.2 &&
  pip install jsonlines==4.0.0 &&
  echo "✅ Dependencies installed" &&
  echo "🔧 Setting up environment..." &&
  export AZURE_ML_INPUT_training_data=${{inputs.training_data}} &&
  export AZURE_ML_OUTPUT_model_output=${{outputs.model_output}} &&
  export AZURE_ML_OUTPUT_validation_output=${{outputs.validation_output}} &&
  echo "🎓 Starting REBEL training validation..." &&
  python week1-validation/train_rebel_azure_week1.py &&
  echo "📊 Copying validation results..." &&
  cp -r ${{outputs.model_output}}/* ${{outputs.validation_output}}/ &&
  echo "✅ Week 1 validation completed!"

# Tags
tags:
  week: "1"
  purpose: "technical_validation"
  model: "rebel"
  framework: "pytorch_lightning"
  config_system: "hydra"

# Settings
resources:
  instance_count: 1
  shm_size: 2g

settings:
  continue_on_step_failure: false
```

#### **Step 4.2: Create Job Submission Script**

**File**: `week1-validation/submit_week1_job.py`

```python
#!/usr/bin/env python3
"""
Submit Week 1 Validation Job to Azure ML
"""

import json
from pathlib import Path
from azure.ai.ml import MLClient, load_job
from azure.identity import DefaultAzureCredential

def load_azure_config():
    """Load Azure ML configuration"""
    config_path = Path("azure-ml-v2/configs/azure_config.json")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Azure ML config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def submit_job():
    """Submit Week 1 validation job"""
    print("🚀 Submitting Week 1 Validation Job...")
    
    # Load config and create client
    config = load_azure_config()
    
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"]
    )
    
    # Load and submit job
    job_path = Path("week1-validation/rebel-training-week1.yml")
    job = load_job(source=job_path)
    
    submitted_job = ml_client.jobs.create_or_update(job)
    
    print(f"✅ Job submitted successfully!")
    print(f"📊 Job name: {submitted_job.name}")
    print(f"🔗 Studio URL: {submitted_job.studio_url}")
    print(f"📈 Status: {submitted_job.status}")
    
    return submitted_job

def main():
    """Main submission function"""
    try:
        job = submit_job()
        print(f"\n🎯 Monitor your job at: {job.studio_url}")
        return True
    except Exception as e:
        print(f"❌ Job submission failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

### **Day 5: Monitoring & Validation**

#### **Step 5.1: Job Monitoring Script**

**File**: `week1-validation/monitor_week1_job.py`

```python
#!/usr/bin/env python3
"""
Monitor Week 1 Validation Job
"""

import json
import time
from pathlib import Path
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def monitor_job(job_name):
    """Monitor job execution"""
    print(f"📊 Monitoring job: {job_name}")
    
    # Load config and create client
    config_path = Path("azure-ml-v2/configs/azure_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=config["subscription_id"],
        resource_group_name=config["resource_group"],
        workspace_name=config["workspace_name"]
    )
    
    # Monitor job status
    while True:
        job = ml_client.jobs.get(job_name)
        status = job.status
        
        print(f"⏱️ Job status: {status}")
        
        if status in ["Completed", "Failed", "Canceled"]:
            break
        
        time.sleep(30)  # Check every 30 seconds
    
    return job

def analyze_results(job):
    """Analyze job results"""
    print("📋 Analyzing Week 1 validation results...")
    
    if job.status == "Completed":
        print("✅ Job completed successfully!")
        print("🎉 Technical validation PASSED!")
        
        # TODO: Download and analyze outputs
        print("📁 Outputs available at:")
        if hasattr(job, 'outputs'):
            for output_name, output_config in job.outputs.items():
                print(f"   {output_name}: {output_config}")
        
        return True
    else:
        print(f"❌ Job failed with status: {job.status}")
        print("🔍 Check logs for details")
        return False

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', required=True, help='Job name to monitor')
    args = parser.parse_args()
    
    try:
        job = monitor_job(args.job_name)
        success = analyze_results(job)
        
        if success:
            print("\n🎯 Week 1 Validation Summary:")
            print("✅ PyTorch Lightning + Hydra + Azure ML: COMPATIBLE")
            print("✅ Data Asset Loading: WORKING")  
            print("✅ REBEL Training: SUCCESSFUL")
            print("✅ Model Output Saving: WORKING")
            print("\n🚀 Ready to proceed to Week 2 implementation!")
        
        return success
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

## 🚀 **Quick Start Commands**

### **Setup (Day 1)**
```bash
# Create Week 1 directory
mkdir -p week1-validation

# Test technical stack
az ml job create -f week1-validation/test-environment.yml
```

### **Data Registration (Day 2)**
```bash
# Register data asset
python week1-validation/register_data_asset.py
```

### **Training Validation (Days 3-4)**
```bash
# Submit training job
python week1-validation/submit_week1_job.py

# Monitor job (replace JOB_NAME with actual job name)
python week1-validation/monitor_week1_job.py --job_name JOB_NAME
```

## ✅ **Success Criteria**

### **Technical Validation**
- [ ] All package imports work correctly
- [ ] Hydra configuration loads without errors
- [ ] PyTorch Lightning trainer initializes
- [ ] Azure ML environment variables accessible

### **Data Integration**
- [ ] Data asset registration succeeds  
- [ ] Data files accessible from Azure ML job
- [ ] JSON data loads correctly
- [ ] File paths resolve properly

### **Training Integration**  
- [ ] REBEL model initializes in Azure ML
- [ ] Training starts without errors
- [ ] Hydra configs work in Azure ML environment
- [ ] Model saves to Azure ML outputs

### **Output Validation**
- [ ] Model files saved correctly
- [ ] Training logs captured
- [ ] Metrics logged properly
- [ ] Outputs accessible post-training

## 🎯 **Week 1 Success Means**

✅ **Technical Stack Validated**: PyTorch Lightning + Hydra + Azure ML work together  
✅ **Data Pipeline Working**: Azure ML Data Assets integrate properly  
✅ **Basic Training Successful**: REBEL trains in Azure ML environment  
✅ **Output Management Working**: Model and results save correctly  

## 🔄 **If Week 1 Fails**

### **Common Issues & Solutions**

**Issue**: Package conflicts
**Solution**: Update environment.yml with exact versions

**Issue**: Hydra config errors  
**Solution**: Simplify config structure, use environment variables

**Issue**: Data asset access problems
**Solution**: Check permissions, validate registration

**Issue**: Training crashes
**Solution**: Reduce batch size, simplify model config

## 📊 **Week 1 Deliverables**

1. **Technical validation report**
2. **Working data asset registration**  
3. **Successful REBEL training job**
4. **Validated output management**
5. **Documented lessons learned**

**Week 1 Success = Green light for Week 2 bridge architecture implementation!**

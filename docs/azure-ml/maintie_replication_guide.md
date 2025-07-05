# MaintIE Replication Guide - Azure ML Implementation

## Executive Summary

This guide provides a complete implementation framework for replicating the MaintIE information extraction project using Azure Machine Learning infrastructure. MaintIE enables automatic entity and relation extraction from maintenance work order texts using dual-model architecture (SpERT + REBEL).

## Project Overview

**MaintIE Framework:**
- **Research Focus**: Information extraction from maintenance short texts
- **Dual Architecture**: Token classification (SpERT) + Sequence-to-sequence (REBEL)
- **Entity Classes**: 5 primary types (Activity, PhysicalObject, Process, Property, State)
- **Relation Types**: 6 categories (contains, hasPart, hasParticipant, hasProperty, isA)
- **Performance Target**: 87-89% F1 entity recognition, 65-72% relation extraction

## Azure Infrastructure Requirements

### Compute Resources
- **Primary Cluster**: Standard_E16s_v3 (16 cores, 128GB RAM)
- **Instance Configuration**: Min 0, Max 2 instances
- **Training Duration**: 18-24 hours per model (CPU-optimized)
- **Resource Group**: azure-ml-uwa
- **Workspace**: azure-ml-uwa-workspace

### Environment Specifications
- **Python Version**: 3.8 (critical requirement)
- **PyTorch**: CPU-optimized configuration
- **Dependencies**: transformers, datasets, scikit-learn
- **Memory Allocation**: High-memory compute for large model processing

## Implementation Architecture

### Phase 1: Infrastructure Setup

```bash
# Create Azure ML compute cluster
az ml compute create --name maintie-cpu-cluster \
  --type amlcompute \
  --size Standard_E16s_v3 \
  --max-instances 2 \
  --min-instances 0 \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace
```

### Phase 2: Environment Configuration

```yaml
# maintie-environment.yml
name: maintie-cpu-env
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.8
  - pytorch
  - transformers
  - datasets
  - scikit-learn
  - pandas
  - numpy
  - spacy
```

### Phase 3: Data Preparation Pipeline

```bash
# Upload MaintIE repository to Azure ML
az ml data create --name maintie-code --type uri_folder --path ./ \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

# Execute dataset creation
python models/create_datasets.py
```

### Phase 4: Model Training Configuration

#### SpERT (Token Classification) Configuration
```yaml
# CPU-optimized training parameters
train_batch_size: 1
eval_batch_size: 2
gradient_accumulation_steps: 32
max_steps: 2000
num_train_epochs: 5.0
learning_rate: 0.00003
precision: 32
```

#### REBEL (Sequence-to-Sequence) Configuration
```yaml
# CPU-optimized training parameters
train_batch_size: 1
eval_batch_size: 2
gradient_acc_steps: 8
max_steps: 1000
num_train_epochs: 3.0
learning_rate: 0.00005
```

## Training Pipeline Implementation

### Azure ML Job Configuration

```yaml
# maintie-training-job.yml
type: command
code: azureml:maintie-code:1
compute: azureml:maintie-cpu-cluster
environment: 
  image: mcr.microsoft.com/azureml/pytorch-1.13-ubuntu20.04-py38-cpu
  conda_file: maintie-environment.yml
command: >
  python models/create_datasets.py &&
  python models/spert/spert.py train --config models/spert/configs/azure-ml/maintie_g_1_train_azureml.conf
experiment_name: maintie-cpu-training
display_name: MaintIE-SpERT-Training
```

### Execution Commands

```bash
# Deploy SpERT training
az ml job create --file maintie-spert-job.yml \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace

# Deploy REBEL training (sequential)
az ml job create --file maintie-rebel-job.yml \
  --resource-group azure-ml-uwa \
  --workspace-name azure-ml-uwa-workspace
```

## Expected Performance Benchmarks

### Entity Recognition (NER) Results
| Entity Class | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| PhysicalObject | 79-84% | 87-90% | 82-84% |
| Activity | 96-98% | 95-99% | 96-98% |
| State | 85-90% | 93-97% | 90-93% |
| Process | 100% | 83-100% | 91-100% |
| Property | 83% | 83% | 83% |

**Overall Performance:**
- **Micro F1**: 87-89%
- **Macro F1**: 89-92%

### Relation Extraction Results
| Relation Type | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| hasPatient | 77-81% | 78-86% | 78-83% |
| hasPart | 61-65% | 64-73% | 63-68% |
| hasAgent | 75-87% | 63-87% | 67-87% |
| isA | 27-35% | 55-80% | 36-48% |
| hasProperty | 33-83% | 20-100% | 23-91% |
| contains | 60-100% | 100% | 75-100% |

**Overall Performance:**
- **Micro F1**: 65-72%
- **Macro F1**: 71-84%

## Monitoring and Evaluation

### Azure ML Studio Tracking
- **Jobs Monitoring**: Azure ML Studio → Jobs → maintie-cpu-training
- **Metrics Tracking**: F1 scores, precision, recall per entity/relation class
- **Resource Utilization**: CPU, memory consumption patterns
- **Training Progress**: Loss curves, validation metrics

### Model Evaluation Framework
```python
# Evaluation script execution
python models/spert/spert.py eval --config models/spert/configs/azure-ml/maintie_g_1_eval_azureml.conf
python models/rebel/src/evaluate.py --model_path ./outputs/maintie_model
```

## Copilot Studio Integration Strategy

### Conversational AI Enhancement through Structured Data Extraction

**Solution Objective**: Transform unstructured maintenance conversations into structured, actionable insights within Copilot Studio workflows.

### Phase 1: Model-to-Endpoint Deployment
**Deploy trained MaintIE models as REST endpoints for real-time conversation enhancement:**

```python
# Azure ML endpoint configuration
from azure.ai.ml import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment

# Register MaintIE model
maintie_model = Model(
    path="./outputs/models/final_model",
    name="maintie-extraction-model",
    description="MaintIE entity and relation extraction for maintenance conversations"
)

# Create managed endpoint
endpoint = ManagedOnlineEndpoint(
    name="maintie-extraction-endpoint",
    description="Real-time maintenance text analysis for Copilot Studio",
    tags={"copilot-integration": "maintenance-extraction"}
)
```

### Phase 2: Copilot Studio Custom Actions Integration
**Enable structured maintenance conversation analysis through custom actions:**

```yaml
# Custom Action: Extract Maintenance Entities
Action Name: ExtractMaintenanceEntities
Description: Process maintenance text to identify entities and relationships
Input Parameters:
  - maintenance_text (string): User input about maintenance issues
  - confidence_threshold (number): Minimum confidence for entity extraction
Output Parameters:
  - entities (array): Extracted maintenance entities
  - relations (array): Identified relationships between entities
  - structured_summary (string): Formatted maintenance summary
```

**Implementation Steps:**
1. **Create Custom Action**: Copilot Studio → Actions → Add action → Web plugin
2. **Configure Endpoint**: Point to deployed MaintIE Azure ML endpoint
3. **Map Input/Output**: Connect conversation variables to extraction results
4. **Test Integration**: Validate entity extraction within conversation flow

### Phase 3: Enhanced Conversation Flows
**Transform maintenance conversations with intelligent entity recognition:**

```yaml
# Conversation Flow Enhancement
Trigger: User mentions maintenance issue
Flow Steps:
  1. Capture user input about maintenance problem
  2. Call ExtractMaintenanceEntities action
  3. Parse extracted entities (Equipment, Activity, State, Property)
  4. Generate structured response based on identified components
  5. Route to appropriate maintenance workflow
```

**Quick-Start Integration Guide:**

**Step 1: Deploy Model Endpoint (30 minutes)**
```bash
# Deploy MaintIE model to Azure ML managed endpoint
az ml online-endpoint create --name maintie-extraction --auth-mode key
az ml online-deployment create --endpoint-name maintie-extraction --model maintie-extraction-model:1
```

**Step 2: Configure Copilot Studio Action (15 minutes)**
1. **Navigate**: Copilot Studio → Topics → System → Conversational boosting
2. **Add Action**: Custom web plugin → Azure ML endpoint URL
3. **Configure Parameters**: Input text field → Output structured entities
4. **Test Extraction**: Validate with sample maintenance text

**Step 3: Implement Conversation Enhancement (20 minutes)**
```yaml
# Topic: Maintenance Issue Analysis
User Input: "The pump motor is overheating and needs inspection"
Action Call: ExtractMaintenanceEntities(maintenance_text: {user_input})
Response Logic:
  - Equipment: pump motor
  - State: overheating  
  - Activity: inspection
  - Generated Response: "I understand you have a pump motor overheating issue requiring inspection. Let me connect you with maintenance scheduling for immediate assessment."
```

**Integration Benefits:**
- **Structured Understanding**: Convert maintenance conversations into actionable data
- **Intelligent Routing**: Direct users to appropriate maintenance workflows based on extracted entities
- **Enhanced Context**: Provide maintenance teams with pre-analyzed conversation summaries
- **Automated Documentation**: Generate structured maintenance reports from conversational input

**Expected Performance**: Sub-2 second response time for real-time conversation enhancement with 87-89% entity recognition accuracy.

## Quick-Start Implementation Guide

### Step 1: Repository Setup (5 minutes)
1. **Clone MaintIE repository**:
   ```bash
   # Clone the original MaintIE repository
   git clone https://github.com/nlp-tlp/maintie.git
   cd maintie
   ```

2. **Create replication branch**:
   ```bash
   # Create and switch to azure-ml-replication branch
   git checkout -b azure-ml-replication
   git push -u origin azure-ml-replication
   ```

3. **Verify repository structure**:
   ```bash
   # Confirm essential directories exist
   ls -la models/spert models/rebel data/
   ```

### Step 2: Infrastructure Deployment (10 minutes)
1. **Create compute cluster**:
   ```bash
   az ml compute create --name maintie-cpu-cluster --type amlcompute --size Standard_E16s_v3 --max-instances 2 --resource-group azure-ml-uwa --workspace-name azure-ml-uwa-workspace
   ```

2. **Verify cluster status**:
   ```bash
   az ml compute show --name maintie-cpu-cluster --resource-group azure-ml-uwa --workspace-name azure-ml-uwa-workspace
   ```

### Step 3: Critical Code Modifications (15 minutes)

**⚠️ Important**: This step implements the complete modifications from `maintie_code_modifications.md` including all 8 scripts and automation tools.

1. **Apply all essential source code fixes**:
   ```bash
   # Execute the complete modification scripts from maintie_code_modifications.md:
   
   # 1. SpERT Configuration Generator - Updates all 8 configs
   python create_azure_configs.py
   
   # 2. REBEL Configuration Updater - Handles Hydra path issues
   python update_rebel_configs.py
   
   # 3. CPU Optimization - Creates CPU-specific training configs
   python optimize_for_cpu.py
   
   # 4. PyTorch Lightning Bug Fix - Critical checkpoint fix
   python fix_pytorch_lightning.py
   
   # 5. Azure Environment Setup - Complete dependency management
   python setup_azure_environment.py
   
   # 6. REBEL Model Download - Automated base model retrieval
   python download_rebel_model.py
   ```

2. **Automated Training Scripts Available**:
   ```bash
   # Complete training entry points (as specified in maintie_code_modifications.md)
   
   # SpERT training with full preprocessing
   python azure_ml_spert_training.py
   
   # REBEL training with model download and config updates
   python azure_ml_rebel_training.py
   ```

3. **Deployment Automation**:
   ```bash
   # Complete Azure ML deployment script
   ./deploy_maintie_azure.sh
   ```

4. **Verify implementation**:
   ```bash
   # Check all required files are created
   ls -la *.py deploy_maintie_azure.sh requirements_azure_ml.txt
   # Note: Azure ML configs will be created when scripts are executed
   ```

3. **Execute all code modifications**:
   ```bash
   # Run all modification scripts from maintie_code_modifications.md
   python create_azure_configs.py
   python update_rebel_configs.py
   python optimize_for_cpu.py
   python fix_pytorch_lightning.py
   python setup_azure_environment.py
   
   # Create comprehensive Azure ML requirements file
   cat > requirements_azure_ml.txt << EOF
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
   EOF
   ```

4. **Commit all critical modifications**:
   ```bash
   git add models/spert/configs/azure-ml/ *.py requirements_azure_ml.txt deploy_maintie_azure.sh
   git commit -m "Complete Azure ML implementation: SpERT configs, REBEL configs, PyTorch Lightning fix, CPU optimization, environment setup"
   ```

5. **Upload modified codebase to Azure ML**:
   ```bash
   az ml data create --name maintie-code-modified --type uri_folder --path ./ --resource-group azure-ml-uwa --workspace-name azure-ml-uwa-workspace
   ```

### Step 3: Training Execution (30 minutes setup + 18-24 hours training)
1. **Create training job**:
   ```bash
   cat > maintie-spert-job.yml << EOF
   type: command
   code: azureml:maintie-code:1
   compute: azureml:maintie-cpu-cluster
   environment: 
     image: mcr.microsoft.com/azureml/pytorch-1.13-ubuntu20.04-py38-cpu
     conda_file: maintie-environment.yml
   command: python models/create_datasets.py && python azure_ml_spert_training.py
   experiment_name: maintie-spert-training
   EOF
   ```

2. **Deploy training pipeline**:
   ```bash
   az ml job create --file maintie-spert-job.yml --resource-group azure-ml-uwa --workspace-name azure-ml-uwa-workspace
   ```

3. **Monitor progress**: Azure ML Studio → Jobs → maintie-spert-training

### Step 4: Model Evaluation (2 hours)
1. **Execute evaluation**:
   ```bash
   python models/spert/spert.py eval --config models/spert/configs/azure-ml/maintie_g_1_eval_azureml.conf
   ```

2. **Validate performance**: Target F1 scores 87-89% entity recognition

### Step 5: Copilot Studio Integration (4-6 hours)
1. **Deploy model endpoints**: Azure ML → Endpoints → Real-time inference
2. **Configure Copilot actions**: Integrate MaintIE APIs into conversation flows
3. **Test integration**: Validate entity extraction within Copilot conversations

## Success Criteria

### Technical Benchmarks
- **Entity Recognition F1**: ≥87%
- **Relation Extraction F1**: ≥65%
- **Training Completion**: Within 24 hours per model
- **Model Deployment**: Successful endpoint creation

### Integration Milestones
- **API Functionality**: MaintIE endpoints responding correctly
- **Copilot Integration**: Successful entity extraction in conversation flows
- **Performance Validation**: Real-time processing under 2 seconds per request

## Git Workflow Management

### Branch Strategy
```bash
# Main development branches
main                    # Original MaintIE codebase
azure-ml-replication   # Azure ML specific modifications
copilot-integration    # Copilot Studio integration features
```

### Version Control Best Practices
```bash
# Track Azure ML specific modifications
git add models/spert/configs/maintie_cpu_*.conf
git commit -m "Add CPU-optimized SpERT configurations for Azure ML"

# Track training progress
git add logs/ outputs/
git commit -m "Training checkpoint: SpERT model epoch 5 - F1: 87.3%"

# Document configuration changes
git add azure-ml-*.yml
git commit -m "Update Azure ML pipeline configuration for Standard_E16s_v3"
```

### Collaborative Development
```bash
# Push replication progress
git push origin azure-ml-replication

# Create pull request for integration
git checkout -b copilot-integration
git push -u origin copilot-integration
```

## Troubleshooting Guide

### Common Issues and Solutions

**Memory Errors**:
- Reduce batch size to 1
- Increase gradient accumulation steps
- Monitor memory usage in Azure ML Studio

**Training Convergence**:
- Adjust learning rate (0.00001-0.0001 range)
- Modify max_steps based on dataset size
- Enable early stopping for optimal convergence

**Azure ML Connection Issues**:
- Verify resource group and workspace names
- Check Azure CLI authentication status
- Validate compute cluster provisioning state

## Resource Optimization

### Cost Management
- **Auto-scaling**: Min instances = 0 for cost efficiency
- **Spot Instances**: Consider for non-critical training runs
- **Monitoring**: Track compute usage through Azure Cost Management

### Performance Tuning
- **Batch Size Optimization**: Balance memory usage vs training speed
- **Parallel Processing**: Utilize multiple CPU cores effectively
- **Memory Management**: Monitor RAM utilization patterns

## Conclusion

This implementation guide provides a complete framework for replicating MaintIE on Azure ML infrastructure. The CPU-optimized configuration ensures immediate deployment capability while maintaining research-grade performance standards. Expected completion timeline: 48-72 hours from infrastructure setup to model deployment, with 87-89% entity recognition accuracy matching published research benchmarks.

For additional support and advanced configuration options, refer to the original MaintIE repository documentation and Azure ML best practices guidelines.
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
python create_datasets.py
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
  python create_datasets.py &&
  python models/spert/spert.py train --config models/spert/configs/maintie_g_1_train.conf
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
python models/spert/spert.py eval --config models/spert/configs/maintie_g_1_eval.conf
python models/rebel/src/evaluate.py --model_path ./outputs/maintie_model
```

## Integration Strategy for Copilot Studio

### Phase 1: Model Deployment
- **Azure ML Endpoints**: Deploy trained models as REST APIs
- **Container Configuration**: Package models for Copilot Studio integration
- **API Interface**: Enable text input → structured entity extraction

### Phase 2: Copilot Studio Integration
- **Custom Actions**: Integrate MaintIE endpoints into Copilot flows
- **Entity Recognition**: Process maintenance texts in conversation flows
- **Knowledge Extraction**: Convert unstructured maintenance data to structured insights

### Phase 3: Conversation Enhancement
- **Maintenance Context**: Enable Copilot to understand maintenance terminology
- **Structured Responses**: Generate responses based on extracted entities/relations
- **Knowledge Base Integration**: Connect extracted information to organizational knowledge

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

### Step 3: Azure ML Configuration (5 minutes)
1. **Create Azure ML configuration files**:
   ```bash
   # Create CPU-optimized environment configuration
   cat > azure-ml-environment.yml << EOF
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
   EOF
   ```

2. **Commit Azure ML configurations**:
   ```bash
   # Add Azure ML specific configurations to branch
   git add azure-ml-environment.yml
   git commit -m "Add Azure ML CPU-optimized environment configuration"
   ```

3. **Upload repository to Azure ML**:
   ```bash
   az ml data create --name maintie-code --type uri_folder --path ./ --resource-group azure-ml-uwa --workspace-name azure-ml-uwa-workspace
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
   command: python create_datasets.py && python models/spert/spert.py train --config models/spert/configs/maintie_g_1_train.conf
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
   python models/spert/spert.py eval --config models/spert/configs/maintie_g_1_eval.conf
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
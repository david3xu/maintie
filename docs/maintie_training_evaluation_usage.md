# MaintIE Model Training, Evaluation & Usage Guide

## Executive Summary

MaintIE is an information extraction system for maintenance work orders that trains specialized NLP models to extract structured data from unstructured maintenance texts. The project employs a comprehensive training and evaluation framework with multiple model architectures and training strategies to find optimal solutions for different complexity requirements.

---

## 🏗️ Architecture Overview

### Core Components

```
MaintIE System Architecture
├── Data Processing Pipeline
│   ├── Raw Maintenance Data (gold_release.json, silver_release.json)
│   ├── Entity Hierarchy Schema (scheme.json - 224 entity types)
│   └── Dataset Generation (4 complexity levels: 1, 5, 32, 224 entity types)
├── Model Training Pipeline
│   ├── SPERT (Token Classification Architecture)
│   └── REBEL (Sequence-to-Sequence Architecture)
├── Evaluation Framework
│   ├── Named Entity Recognition (NER) Metrics
│   ├── Relation Extraction (RE) Metrics
│   └── Cross-Architecture Performance Comparison
└── Production Deployment
    ├── Model Selection Strategy
    ├── Inference Pipeline
    └── Structured Output Generation
```

---

## 📊 Data Architecture & Preprocessing

### Entity Hierarchy Design

**Multi-Level Complexity Testing:**
- **Level 0**: 1 entity type (generic entity detection)
- **Level 1**: 5 entity types (PhysicalObject, Activity, Process, State, Property)
- **Level 2**: 32 entity types (mid-level hierarchy)
- **Level 3**: 224 entity types (full fine-grained classification)

### Dataset Variants

```
Dataset Generation Process:
├── create_datasets.py
├── Input: Real MaintIE corpus (1,076 expert-annotated texts)
├── Output: 8 dataset variants
│   ├── Gold Corpus (g-0, g-1, g-2, g-3) - Expert annotations
│   └── Silver Corpus (s-0, s-1, s-2, s-3) - Model-generated annotations
```

**Data Quality Levels:**
- **Gold (Fine-grained)**: Expert-annotated, high-quality, 1,076 texts
- **Silver (Coarse-grained)**: Model-generated, large-scale, 7,000 texts

---

## 🤖 Model Training Architecture

### Model Architectures

#### SPERT (Span-based Entity and Relation Transformer)
```
SPERT Architecture:
├── Base Model: BERT-base-cased
├── Task Type: Token classification with span detection
├── Training Method: Joint entity and relation extraction
├── Strengths: Fast inference, explicit span boundaries
├── Use Case: Production systems requiring interpretable extractions
└── Output: Spans with entity types + relation triplets
```

#### REBEL (Relation Extraction By End-to-end Language generation)
```
REBEL Architecture:
├── Base Model: BART/T5-based sequence-to-sequence
├── Task Type: Text generation
├── Training Method: Generate structured text containing entities and relations
├── Strengths: Flexible output format, handles complex relation structures
├── Use Case: Research applications with complex relation schemas
└── Output: Generated text with embedded entity-relation structures
```

### Training Strategies

#### 1. Direct Fine-tuning (FG)
```
Direct Fine-tuning Process:
├── Input: Pre-trained BERT/REBEL-base model
├── Training Data: Target complexity level (gold corpus)
├── Process: Single-stage fine-tuning directly on target task
├── Purpose: Baseline performance measurement
└── Research Question: How well do pre-trained models adapt to domain hierarchies?
```

#### 2. Sequential Fine-tuning (CG+FG)
```
Sequential Fine-tuning Process:
├── Stage 1: Pre-fine-tuning on Silver Corpus (coarse-grained)
│   ├── Input: Pre-trained BERT/REBEL-base
│   ├── Training: Large-scale silver corpus (7,000 texts)
│   └── Output: Domain-adapted base model
├── Stage 2: Fine-tuning on Gold Corpus (fine-grained)
│   ├── Input: Domain-adapted model from Stage 1
│   ├── Training: High-quality gold corpus (1,076 texts)
│   └── Output: Final production model
├── Purpose: Test curriculum learning hypothesis
└── Research Question: Does intermediate training improve final performance?
```

### Complete Training Matrix

**16 Total Training Experiments:**

| Model | Strategy | Complexity | Experiment | Training Path |
|-------|----------|------------|------------|---------------|
| SPERT | Direct | 1 type | FG-0 | BERT-base → Gold-0 |
| SPERT | Direct | 5 types | FG-1 | BERT-base → Gold-1 |
| SPERT | Direct | 32 types | FG-2 | BERT-base → Gold-2 |
| SPERT | Direct | 224 types | FG-3 | BERT-base → Gold-3 |
| SPERT | Sequential | 1 type | CG+FG-0 | BERT-base → Silver-0 → Gold-0 |
| SPERT | Sequential | 5 types | CG+FG-1 | BERT-base → Silver-1 → Gold-1 |
| SPERT | Sequential | 32 types | CG+FG-2 | BERT-base → Silver-2 → Gold-2 |
| SPERT | Sequential | 224 types | CG+FG-3 | BERT-base → Silver-3 → Gold-3 |
| REBEL | Direct | 1 type | REBEL-FG-0 | REBEL-base → Gold-0 |
| REBEL | Direct | 5 types | REBEL-FG-1 | REBEL-base → Gold-1 |
| REBEL | Direct | 32 types | REBEL-FG-2 | REBEL-base → Gold-2 |
| REBEL | Direct | 224 types | REBEL-FG-3 | REBEL-base → Gold-3 |
| REBEL | Sequential | 1 type | REBEL-CG+FG-0 | REBEL-base → Silver-0 → Gold-0 |
| REBEL | Sequential | 5 types | REBEL-CG+FG-1 | REBEL-base → Silver-1 → Gold-1 |
| REBEL | Sequential | 32 types | REBEL-CG+FG-2 | REBEL-base → Silver-2 → Gold-2 |
| REBEL | Sequential | 224 types | REBEL-CG+FG-3 | REBEL-base → Silver-3 → Gold-3 |

---

## 📈 Evaluation Framework

### Evaluation Metrics

#### Named Entity Recognition (NER)
```
NER Evaluation Metrics:
├── Micro F1-Score: Overall performance across all entity types
├── Macro F1-Score: Average performance per entity type
├── Precision: Correctly identified entities / Total predicted entities
├── Recall: Correctly identified entities / Total actual entities
└── Entity Type Distribution: Performance breakdown by entity class
```

#### Relation Extraction (RE)
```
RE Evaluation Metrics:
├── Strict Evaluation: Exact entity spans and relation types must match
├── Loose Evaluation: Overlapping entity spans with correct relation types
├── Micro F1-Score: Overall relation extraction performance
├── Macro F1-Score: Average performance per relation type
└── Relation Type Analysis: Performance breakdown by relation class
```

### Evaluation Process

#### Training-Time Evaluation
```
Training Evaluation:
├── Validation Set: Used during training for early stopping
├── Metrics Logged: NER F1, RE F1, Loss values
├── Checkpointing: Save best performing models
└── Progress Tracking: Monitor convergence and overfitting
```

#### Final Model Evaluation
```
Test Set Evaluation:
├── Hold-out Test Set: Final evaluation on unseen data
├── Comprehensive Metrics: Full NER and RE performance analysis
├── Error Analysis: Identify common failure patterns
├── Cross-Architecture Comparison: SPERT vs REBEL performance
└── Complexity Impact Analysis: Performance vs entity hierarchy depth
```

### Benchmark Validation

**Expected Performance Ranges (Level 1 - 5 entity types):**

| Metric | SPERT Performance | REBEL Performance |
|--------|------------------|-------------------|
| NER Micro F1 | 87-90% | 85-88% |
| RE Strict F1 | 72-75% | 70-73% |
| RE Loose F1 | 74-77% | 72-75% |
| Training Time | ~20 min (CPU) | ~30 min (GPU) |
| Inference Speed | Fast | Moderate |

---

## 🚀 Production Usage Process

### Model Selection Strategy

#### 1. Requirements Analysis
```
Usage Requirements Assessment:
├── Performance Priority: Accuracy vs Speed
├── Complexity Needs: Entity granularity requirements
├── Infrastructure: CPU vs GPU availability
├── Integration: Real-time vs batch processing
└── Interpretability: Explainable vs black-box acceptance
```

#### 2. Model Architecture Selection
```
Architecture Decision Matrix:
├── Production Systems → SPERT
│   ├── Fast inference (token classification)
│   ├── Explicit span boundaries
│   ├── Lower computational requirements
│   └── Easier integration with existing NLP pipelines
├── Research Applications → REBEL
│   ├── Flexible output format
│   ├── Complex relation handling
│   ├── Generative capabilities
│   └── Better handling of unseen relation types
```

#### 3. Complexity Level Selection
```
Complexity Level Decision:
├── Level 1 (5 types): Basic maintenance categorization
│   ├── Use Case: High-level asset management
│   ├── Performance: Highest accuracy
│   └── Example: Equipment type classification
├── Level 2 (32 types): Moderate detail extraction
│   ├── Use Case: Maintenance activity tracking
│   ├── Performance: Balanced accuracy/granularity
│   └── Example: Specific maintenance procedures
├── Level 3 (224 types): Fine-grained analysis
│   ├── Use Case: Detailed engineering analysis
│   ├── Performance: Lower accuracy, maximum detail
│   └── Example: Component-level failure analysis
```

### Deployment Architecture

#### Inference Pipeline
```
Production Inference Pipeline:
├── Input Processing
│   ├── Text Preprocessing: Tokenization, normalization
│   ├── Input Validation: Format checking, length limits
│   └── Batch Formation: Efficient processing grouping
├── Model Inference
│   ├── Model Loading: Pre-trained model initialization
│   ├── Feature Extraction: Context encoding
│   └── Prediction Generation: Entity and relation extraction
├── Output Processing
│   ├── Post-processing: Confidence filtering, duplicate removal
│   ├── Format Conversion: JSON/XML structured output
│   └── Validation: Consistency checking
└── Integration
    ├── API Endpoints: RESTful service interfaces
    ├── Database Integration: Structured data storage
    └── Monitoring: Performance and error tracking
```

#### Real-World Usage Example

```
Example Usage Flow:
├── Input: "pump motor failed, technician replaced bearing"
├── Preprocessing: Tokenization and normalization
├── Model Inference: SPERT Level 1 model
├── Output Extraction:
│   ├── Entities:
│   │   ├── "pump motor" → PhysicalObject
│   │   ├── "failed" → State
│   │   ├── "technician" → PhysicalObject/Person
│   │   ├── "replaced" → Activity
│   │   └── "bearing" → PhysicalObject
│   └── Relations:
│       ├── pump motor ← hasParticipant/hasPatient ← failed
│       ├── technician ← hasParticipant/hasAgent ← replaced
│       └── bearing ← hasParticipant/hasPatient ← replaced
├── Structured Output: JSON with entities and relations
└── Integration: Store in maintenance management system
```

### Integration Patterns

#### Enterprise Asset Management Integration
```
Enterprise Integration:
├── CMMS Integration: Direct integration with maintenance systems
├── IoT Data Fusion: Combine with sensor data for comprehensive analysis
├── Business Intelligence: Feed structured data to analytics platforms
├── Workflow Automation: Trigger maintenance procedures based on extracted information
└── Knowledge Base Building: Accumulate maintenance knowledge over time
```

#### API Design Pattern
```
REST API Design:
├── POST /extract/entities
│   ├── Input: {"text": "maintenance work order text"}
│   ├── Output: {"entities": [...], "relations": [...], "confidence": 0.95}
├── POST /extract/batch
│   ├── Input: {"texts": ["text1", "text2", ...]}
│   ├── Output: {"results": [{"entities": [...], "relations": [...]}, ...]}
├── GET /models/status
│   ├── Output: {"model": "SPERT-Level1", "status": "ready", "version": "1.0"}
└── GET /health
    ├── Output: {"status": "healthy", "latency": "50ms"}
```

---

## 🔧 Technical Implementation Details

### Training Infrastructure Requirements

#### CPU-Based Training (Demonstrated Feasible)
```
CPU Training Configuration:
├── Hardware: 16+ cores, 128GB+ RAM
├── Training Time: 15-20 minutes per epoch
├── Memory Usage: 8-12GB peak
├── Throughput: 2-3 batches/second
└── Use Case: Development and smaller datasets
```

#### GPU-Based Training (Recommended for Production)
```
GPU Training Configuration:
├── Hardware: V100/A100 class GPUs
├── Training Time: 2-5 minutes per epoch
├── Memory Usage: 16-24GB GPU memory
├── Throughput: 10-20 batches/second
└── Use Case: Large-scale production training
```

### Model Artifacts

#### Trained Model Components
```
Model Artifact Structure:
├── pytorch_model.bin: Trained model weights
├── config.json: Model configuration parameters
├── tokenizer files: Text processing components
├── training_args.json: Training hyperparameters
├── metrics.json: Evaluation results
└── label mappings: Entity and relation type mappings
```

---

## 📊 Performance Analysis Framework

### Research Questions Addressed

1. **Hierarchy Complexity Impact**: How does increasing entity type granularity affect model performance?
2. **Training Strategy Effectiveness**: Does sequential fine-tuning improve over direct fine-tuning?
3. **Architecture Comparison**: Which model architecture performs better for maintenance domain tasks?
4. **Practical Deployment Trade-offs**: What are the optimal complexity levels for different use cases?

### Key Findings

#### Hierarchy Complexity Impact
- **Clear performance degradation** as entity types increase from 5 → 32 → 224
- **Relation extraction more sensitive** to complexity than named entity recognition
- **Level 1 (5 types) optimal** for performance/granularity balance

#### Sequential vs Direct Training
- **Sequential training benefits** vary by complexity level
- **Most effective** for moderate complexity levels (32 entity types)
- **Diminishing returns** at highest complexity (224 types)

#### Architecture Performance
- **SPERT**: Better for production deployment (speed, interpretability)
- **REBEL**: Better for complex relation extraction research

---

## 🎯 Deployment Decision Matrix

### Selection Criteria

| Requirement | Recommended Solution | Rationale |
|-------------|---------------------|-----------|
| **High-volume production** | SPERT Level 1 | Fast inference, proven reliability |
| **Detailed failure analysis** | SPERT Level 2 | Balanced detail/performance |
| **Research applications** | REBEL Level 1/2 | Flexible output, complex relations |
| **Real-time processing** | SPERT Level 1 | Lowest latency |
| **Maximum detail extraction** | SPERT/REBEL Level 3 | Full entity hierarchy |
| **CPU-only deployment** | SPERT any level | CPU-optimized training proven |
| **Integration with Azure** | SPERT Level 1 | Best enterprise integration path |

### Success Metrics for Production

#### Technical Metrics
- **Accuracy**: NER F1 > 85%, RE F1 > 70%
- **Latency**: < 100ms per work order
- **Throughput**: > 1000 work orders/minute
- **Availability**: 99.9% uptime

#### Business Metrics
- **Data Quality**: 90% reduction in manual annotation time
- **Process Efficiency**: 50% faster maintenance categorization
- **Knowledge Extraction**: 80% of critical maintenance information captured
- **ROI**: Positive return within 6 months of deployment

---

This comprehensive framework provides the foundation for successful deployment of MaintIE in production environments while maintaining research reproducibility and enabling continuous improvement of maintenance information extraction capabilities.
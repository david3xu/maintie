# MaintIE Data Splits: Train/Validation/Test Explained
## Understanding How Machine Learning Data is Split and Used

**Context**: MaintIE Project Data Usage Strategy  
**Purpose**: Explain the three-way data split and its crucial role in reliable model evaluation

---

## 🎯 **The Three-Way Split: A Car Driving Analogy**

Imagine you're learning to drive a car. The three data splits work like different stages of driver training:

### **Training Set = Driving Lessons with Instructor**
- **Purpose**: Learning the basic skills
- **Feedback**: Immediate correction and guidance
- **Goal**: Build fundamental competence

### **Validation Set = Practice Test with Instructor**
- **Purpose**: Check progress without teaching new skills
- **Feedback**: Performance assessment to identify weak areas
- **Goal**: Fine-tune skills before the real test

### **Test Set = Official Driving Test**
- **Purpose**: Unbiased assessment of real-world capability
- **Feedback**: Pass/fail - no teaching or hints
- **Goal**: Prove you can drive safely without assistance

---

## 📊 **MaintIE Data Split Statistics**

### **Dataset Overview**
```
Total MaintIE Gold Corpus: 1,076 maintenance texts
├── Training Set:    860 texts (79.9%) 
├── Validation Set:  108 texts (10.0%)
└── Test Set:        108 texts (10.0%)
```

### **Why These Proportions?**

**80/10/10 Split** is a common machine learning practice because:
- **80% Training**: Enough data for models to learn patterns
- **10% Validation**: Sufficient for reliable performance estimation  
- **10% Test**: Independent final evaluation without overfitting

### **Per-Level Data Distribution**

Each complexity level maintains the same split:

| **Level** | **Entity Types** | **Train** | **Validation** | **Test** | **Total** |
|-----------|------------------|-----------|----------------|----------|-----------|
| **FG-0** | 1 type | 860 | 108 | 108 | 1,076 |
| **FG-1** | 5 types | 860 | 108 | 108 | 1,076 |
| **FG-2** | 32 types | 860 | 108 | 108 | 1,076 |
| **FG-3** | 224 types | 860 | 108 | 108 | 1,076 |

**Key Point**: The **same text examples** appear in all levels, but with different annotation granularity.

---

## 🔄 **Complete Workflow: How Each Split is Used**

### **Phase 1: Model Training (Using Training Set)**

#### **What Happens**
```
For each training epoch:
1. Model sees training examples one by one
2. Makes predictions on entities and relations  
3. Compares predictions to true annotations
4. Updates internal parameters to reduce errors
5. Repeats for all 860 training texts
```

#### **Example Training Process**
```
Training Text: "Replace faulty pressure sensor in cooling system"
True Labels: 
- "Replace" → Activity/MaintenanceActivity/Replace
- "pressure sensor" → PhysicalObject/SensingObject/PressureSensingObject
- "cooling system" → PhysicalObject/EmittingObject/ThermalCoolingObject
- Relation: "Replace" hasPatient "pressure sensor"

Model Prediction: 
- "Replace" → Activity/MaintenanceActivity/Replace ✅
- "pressure sensor" → PhysicalObject/SensingObject ❌ (too general)
- "cooling system" → PhysicalObject ❌ (way too general)
- Relation: "Replace" hasPatient "pressure sensor" ✅

Model Learning: Adjust weights to make more specific predictions
```

#### **Training Set Purpose**
- **Teach the model** what maintenance entities and relations look like
- **Learn patterns** from 860 diverse maintenance work orders
- **Build internal representations** of technical language

### **Phase 2: Validation During Training (Using Validation Set)**

#### **What Happens** 
```
After each training epoch:
1. Model makes predictions on validation set (108 texts)
2. Predictions compared to true annotations
3. Performance metrics calculated (F1, Precision, Recall)
4. Results used to decide if training should continue
5. NO model parameters updated based on validation
```

#### **Validation Set Purpose**
- **Monitor training progress** without biasing the model
- **Detect overfitting**: When training improves but validation doesn't
- **Early stopping**: Stop training when validation performance plateaus
- **Hyperparameter tuning**: Compare different learning rates, batch sizes, etc.

#### **Example Validation Monitoring**
```
Epoch 1: Train F1=65%, Validation F1=60% → Keep training
Epoch 2: Train F1=75%, Validation F1=68% → Keep training  
Epoch 3: Train F1=85%, Validation F1=72% → Keep training
Epoch 4: Train F1=90%, Validation F1=71% → Potential overfitting
Epoch 5: Train F1=95%, Validation F1=69% → Stop training!
```

### **Phase 3: Final Evaluation (Using Test Set)**

#### **What Happens**
```
After training is completely finished:
1. Model makes predictions on test set (108 texts)
2. Predictions compared to true annotations
3. Final performance metrics calculated
4. Results published as model's true capability
5. Model NEVER sees test set during training
```

#### **Test Set Purpose**
- **Unbiased performance estimate**: Model has never seen these examples
- **Real-world simulation**: How well will model work on new maintenance texts?
- **Publication results**: Official numbers reported in research papers
- **Model comparison**: Fair comparison between different approaches

---

## 🚨 **Critical Rules: Why the Splits Must Stay Separate**

### **Rule 1: Never Train on Validation/Test Data**

**Why**: If the model sees the "exam questions" during training, it will memorize rather than learn general patterns.

**Bad Example**:
```
❌ WRONG: Use validation set for training
→ Model memorizes validation examples
→ Artificially high validation scores
→ Overly optimistic performance estimates
```

**Good Example**:
```
✅ CORRECT: Keep validation separate
→ Model learns general patterns from training
→ Validation provides honest performance estimate
→ Reliable model development
```

### **Rule 2: Never Use Test Set for Model Development**

**Why**: Any decisions based on test performance will bias the results.

**Bad Example**:
```
❌ WRONG: Check test set and adjust model
→ Try 10 different architectures
→ Pick the one with best test performance  
→ "Test" performance is no longer unbiased
```

**Good Example**:
```
✅ CORRECT: Use validation for all decisions
→ Try 10 different architectures
→ Pick the one with best validation performance
→ Test set provides true unbiased evaluation
```

### **Rule 3: Test Set is Used ONLY ONCE**

**Why**: Multiple test evaluations lead to indirect overfitting.

**Process**:
```
1. Design model using training + validation
2. Finalize all decisions
3. Run test evaluation ONCE
4. Report results (good or bad!)
```

---

## 📈 **MaintIE Specific Implementation**

### **Training Logs Show the Split in Action**

From the replication logs, we can see exactly how MaintIE uses the splits:

```
Parse dataset 'train': 100%|█| 860/860 [00:00<00:00, 1597/s]
Parse dataset 'valid': 100%|█| 108/108 [00:00<00:00, 1850/s]

Training: 860 documents, 2716 entities, 1881 relations
Validation: 108 documents, 334 entities, 226 relations
```

### **Validation During Training**
```
Train epoch 0: 100%|███████████| 860/860 [06:23<00:00, 2.24it/s]
Evaluate: valid
Evaluate epoch 1: 100%|████████| 54/54 [00:08<00:00, 6.45it/s]

Evaluation Results:
--- Entities (NER) ---
Entity precision=73.41%, recall=90.12%, f1-score=80.91%

--- Relations ---  
micro precision=40.23%, recall=47.35%, f1-score=43.50%
```

### **File Structure Shows the Splits**
```
maintie/models/data/g-1/
├── maintie_train.json     # 860 training examples
├── maintie_dev.json       # 108 validation examples  
└── maintie_test.json      # 108 test examples
```

---

## 🔬 **Why This Matters for MaintIE Research**

### **Reliable Performance Estimates**

The three-way split ensures that MaintIE results are **trustworthy**:

**What we can trust**:
- Training performance → Model can learn maintenance patterns
- Validation performance → Model generalizes beyond training examples
- Test performance → Model will work on new, unseen maintenance texts

### **Fair Model Comparison**

Because all models use the **same test set**, we can fairly compare:
- **SPERT vs REBEL**: Which architecture works better?
- **Direct vs Sequential**: Which training strategy is superior?
- **FG-1 vs FG-2**: Where is the complexity sweet spot?

### **Real-World Deployment Confidence**

Test set performance predicts how the model will work when deployed:
- **High test F1-score** → Confident about production deployment
- **Large train/test gap** → Model may not generalize well
- **Consistent across complexity levels** → Robust approach

---

## 🎯 **Practical Implications for Users**

### **For Researchers**
```
✅ DO:
- Use validation set for hyperparameter tuning
- Report test set results as final performance
- Compare models using same test set
- Keep test set completely separate during development

❌ DON'T:
- Make any decisions based on test set performance
- Use test set multiple times
- Mix training and validation data
- Report validation performance as final results
```

### **For Practitioners**
```
✅ Understanding Performance:
- Training performance = Model learning capability
- Validation performance = Expected real-world performance
- Test performance = Unbiased capability assessment

✅ Deployment Decisions:
- Use test F1-scores for deployment planning
- Expect performance similar to test set results
- Plan for slight degradation in production
```

### **For Domain Experts**
```
✅ Trust Levels:
- Test results = Reliable estimates for your maintenance texts
- Validation results = Good estimates during development
- Training results = Optimistic estimates (model has seen this data)

✅ Decision Making:
- Use test performance for ROI calculations
- Plan annotation budget based on validation curves
- Set realistic expectations based on test results
```

---

## 📊 **Summary: The Three-Set Strategy Success**

### **Training Set (860 texts)**
- **Purpose**: Teach the model maintenance language patterns
- **Usage**: Model sees these repeatedly during training
- **Analogy**: Study materials for an exam

### **Validation Set (108 texts)**  
- **Purpose**: Monitor learning progress and tune hyperparameters
- **Usage**: Check performance after each training epoch
- **Analogy**: Practice exam to gauge readiness

### **Test Set (108 texts)**
- **Purpose**: Provide unbiased final performance assessment
- **Usage**: Single evaluation after training is complete
- **Analogy**: Final exam that determines graduation

### **Why This Works**
1. **Prevents overfitting**: Model can't memorize evaluation data
2. **Enables fair comparison**: All models evaluated on same unseen data
3. **Provides reliable estimates**: Test performance predicts real-world capability
4. **Supports scientific rigor**: Results are reproducible and trustworthy

The three-way split is **fundamental to machine learning credibility** - it's the difference between a model that works in the lab and one that works in the real world of industrial maintenance.

---

*This rigorous data splitting strategy is why MaintIE results can be trusted for making real-world deployment decisions in industrial maintenance applications.*
#!/usr/bin/env python3
"""
🏆 COMPREHENSIVE MODEL EVALUATION SCRIPT
Evaluates all REBEL and SpERT models and produces results in RESULTS.md format

This script:
1. Evaluates REBEL (sequence-to-sequence) models on all hierarchy levels
2. Evaluates SpERT (token classification) models on all hierarchy levels
3. Produces both NER and RE (strict/loose) evaluation metrics
4. Formats output exactly like RESULTS.md
5. Handles both Fine-Grained (FG) and Coarse-Grained→Fine-Grained (CG+FG) experiments
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tempfile
import pandas as pd
from collections import defaultdict
import argparse

# Add models to path for imports
sys.path.append('models/rebel/src')

class ComprehensiveEvaluator:
    """Comprehensive evaluation manager for all MaintIE experiments"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.data_dir = self.base_dir / "models" / "data"
        self.rebel_dir = self.base_dir / "models" / "rebel"
        self.spert_dir = self.base_dir / "models" / "spert"

        # Entity hierarchy mapping
        self.hierarchy_map = {
            "0": {"classes": 1, "name": "Untyped"},
            "1": {"classes": 5, "name": "5 Classes"},
            "2": {"classes": 32, "name": "32 Classes"},
            "3": {"classes": 224, "name": "224 Classes"}
        }

        # Experiment configurations
        self.experiments = {
            "REBEL": {
                "FG": {  # Fine-Grained (Direct training)
                    "0": "rebel_fg_0",
                    "1": "rebel_fg_1",
                    "2": "rebel_fg_2",
                    "3": "rebel_fg_3"
                },
                "CG+FG": {  # Coarse-Grained → Fine-Grained (Sequential)
                    "0": "rebel_cgfg_0",
                    "1": "rebel_cgfg_1",
                    "2": "rebel_cgfg_2",
                    "3": "rebel_cgfg_3"
                }
            },
            "SpERT": {
                "FG": {
                    "0": "maintie_g_0_train",
                    "1": "maintie_g_1_train",
                    "2": "maintie_g_2_train",
                    "3": "maintie_g_3_train"
                },
                "CG+FG": {
                    "0": "maintie_gs_0_train",
                    "1": "maintie_gs_1_train",
                    "2": "maintie_gs_2_train",
                    "3": "maintie_gs_3_train"
                }
            }
        }

        # Standard relation types for MaintIE
        self.relation_types = [
            "contains", "isA", "hasPatient", "hasProperty",
            "hasAgent", "hasPart"
        ]

        # Entity types by hierarchy level
        self.entity_types = {
            "1": ["Entity"],
            "5": ["PhysicalObject", "Activity", "State", "Process", "Property"],
            "32": [
                "Property/UndesirableProperty", "Activity/SupportingActivity",
                "Activity/MaintenanceActivity", "PhysicalObject/StoringObject",
                "PhysicalObject/HoldingObject", "PhysicalObject/Substance",
                "PhysicalObject/RestrictingObject", "PhysicalObject/InterfacingObject",
                "PhysicalObject/PresentingObject", "PhysicalObject/ProtectingObject",
                "PhysicalObject/MatterProcessingObject", "State/UndesirableState",
                "Process/UndesirableProcess", "PhysicalObject/DrivingObject",
                "PhysicalObject/GuidingObject", "PhysicalObject/InformationProcessingObject",
                "PhysicalObject/Organism", "PhysicalObject/EmittingObject",
                "PhysicalObject/ControllingObject", "PhysicalObject/SensingObject",
                "Property", "PhysicalObject/GeneratingObject", "PhysicalObject",
                "PhysicalObject/TransformingObject", "PhysicalObject/CoveringObject"
            ],
            "224": []  # Too many to list, will be loaded dynamically
        }

    def load_entity_types_from_data(self, level: str) -> List[str]:
        """Load entity types from dataset"""
        types_file = self.data_dir / f"g-{level}" / "maintie_types.json"
        if types_file.exists():
            with open(types_file, 'r') as f:
                types_data = json.load(f)
                return [ent["short"] for ent in types_data.get("entities", [])]
        return self.entity_types.get(level, [])

    def evaluate_rebel_model(self, level: str, experiment_type: str) -> Dict[str, Any]:
        """Evaluate a REBEL model using the official evaluation script"""
        print(f"🤖 Evaluating REBEL {experiment_type}-{level}...")

        # Determine checkpoint path based on experiment type
        if experiment_type == "FG":
            # Direct fine-tuning checkpoints
            if level == "1":
                checkpoint_dir = "experiments/rebel_final_corrected"
            else:
                # Look for other FG experiment checkpoints
                checkpoint_dir = f"experiments/rebel_fg_{level}"
        else:  # CG+FG
            checkpoint_dir = f"experiments/rebel_cgfg_{level}"

        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"   ⚠️  Checkpoint not found: {checkpoint_path}")
            return {"error": f"Checkpoint not found: {checkpoint_path}"}

        # Find the actual checkpoint file
        ckpt_files = list(checkpoint_path.glob("*.ckpt"))
        if not ckpt_files:
            print(f"   ⚠️  No .ckpt files found in {checkpoint_path}")
            return {"error": f"No checkpoint files found"}

        ckpt_file = ckpt_files[0]  # Use the first checkpoint found

        # Run the official evaluation
        try:
            cmd = [
                "python", "official_rebel_evaluation.py",
                "--checkpoint_path", str(ckpt_file),
                "--hierarchy_level", level,
                "--output_format", "json"
            ]

            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse the JSON output
            evaluation_results = json.loads(result.stdout)
            return evaluation_results

        except subprocess.CalledProcessError as e:
            print(f"   ❌ Evaluation failed: {e}")
            print(f"   STDERR: {e.stderr}")
            return {"error": f"Evaluation failed: {e}"}
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse evaluation output: {e}")
            return {"error": f"JSON parsing failed: {e}"}

    def evaluate_spert_model(self, level: str, experiment_type: str) -> Dict[str, Any]:
        """Evaluate a SpERT model"""
        print(f"🎯 Evaluating SpERT {experiment_type}-{level}...")

        # Determine model path
        if experiment_type == "FG":
            model_name = f"maintie_g_{level}_train"
        else:  # CG+FG
            model_name = f"maintie_gs_{level}_train"

        model_path = self.spert_dir / "data" / "save" / model_name
        if not model_path.exists():
            print(f"   ⚠️  Model not found: {model_path}")
            return {"error": f"Model not found: {model_path}"}

        # Run SpERT evaluation
        try:
            eval_config = f"maintie_g_{level}_eval.conf"

            cmd = [
                "python", "spert.py", "eval",
                "--config", f"configs/{eval_config}",
                "--model", str(model_path)
            ]

            result = subprocess.run(
                cmd,
                cwd=self.spert_dir,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse SpERT output (implement parsing logic based on SpERT output format)
            return self.parse_spert_output(result.stdout, level)

        except subprocess.CalledProcessError as e:
            print(f"   ❌ SpERT evaluation failed: {e}")
            return {"error": f"SpERT evaluation failed: {e}"}

    def parse_spert_output(self, output: str, level: str) -> Dict[str, Any]:
        """Parse SpERT evaluation output"""
        # This would need to be implemented based on the actual SpERT output format
        # For now, return a placeholder structure
        return {
            "ner": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "per_class": {}
            },
            "re_strict": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "per_class": {}
            },
            "re_loose": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
                "per_class": {}
            }
        }

    def format_classification_report(self, results: Dict[str, Any], title: str) -> str:
        """Format results in classification report style"""
        output = [f"\n{title}\n"]
        output.append("```")

        # Header
        output.append(f"{'type':>20s}    {'precision':>9s}       {'recall':>6s}     {'f1-score':>8s}      {'support':>7s}")

        # Per-class results
        if "per_class" in results:
            for class_name, metrics in results["per_class"].items():
                output.append(f"{class_name:>20s}        {metrics['precision']:>6.2f}        {metrics['recall']:>6.2f}        {metrics['f1']:>6.2f}          {metrics['support']:>3d}")

        # Micro and macro averages
        output.append("")
        output.append(f"{'micro':>20s}        {results['precision']:>6.2f}        {results['recall']:>6.2f}        {results['f1']:>6.2f}          {results['support']:>3d}")
        output.append(f"{'macro':>20s}        {results.get('macro_precision', 0.0):>6.2f}        {results.get('macro_recall', 0.0):>6.2f}        {results.get('macro_f1', 0.0):>6.2f}          {results['support']:>3d}")

        output.append("```")
        return "\n".join(output)

    def generate_results_section(self, model_name: str, experiment_type: str) -> str:
        """Generate a complete results section for a model and experiment type"""
        output = [f"\n### {model_name} ({experiment_type})\n"]

        for level, info in self.hierarchy_map.items():
            output.append(f"\n##### Entity Classes: {info['classes']}\n")

            # Evaluate the model
            if model_name == "REBEL":
                results = self.evaluate_rebel_model(level, experiment_type)
            else:  # SpERT
                results = self.evaluate_spert_model(level, experiment_type)

            if "error" in results:
                output.append(f"⚠️ Error evaluating {model_name} {experiment_type}-{level}: {results['error']}\n")
                continue

            # Format NER results (only for SpERT)
            if model_name == "SpERT" and "ner" in results:
                output.append(self.format_classification_report(results["ner"], "NER"))

            # Format RE results
            if "re_strict" in results:
                output.append(self.format_classification_report(results["re_strict"], "Strict RE"))

            if "re_loose" in results:
                output.append(self.format_classification_report(results["re_loose"], "Loose RE"))

        return "\n".join(output)

    def generate_complete_results_document(self) -> str:
        """Generate the complete results document in RESULTS.md format"""
        output = []

        # Header
        output.append("# Complete Evaluation Results")
        output.append("\nThis document presents the detailed results of entity and relation extraction models.")
        output.append("Results are obtained from the test set portion of the fine-grained expert-annotated corpus.")

        # SpERT Results
        output.append("\n## SpERT (Token Classification)")
        output.append(self.generate_results_section("SpERT", "Fine-Grained"))
        output.append(self.generate_results_section("SpERT", "Coarse-Grained → Fine-Grained"))

        # REBEL Results
        output.append("\n## REBEL (Sequence-to-Sequence)")
        output.append(self.generate_results_section("REBEL", "Fine-Grained"))
        output.append(self.generate_results_section("REBEL", "Coarse-Grained → Fine-Grained"))

        return "\n".join(output)

    def run_complete_evaluation(self, output_file: str = "EVALUATION_RESULTS.md"):
        """Run complete evaluation of all models and generate results document"""
        print("🏆 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 50)
        print("Evaluating all REBEL and SpERT models...")
        print("This may take some time...")

        # Generate complete results
        results_content = self.generate_complete_results_document()

        # Write to output file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write(results_content)

        print(f"\n✅ Complete evaluation results saved to: {output_path}")
        print(f"📊 Results formatted in RESULTS.md style")

        return output_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive MaintIE Model Evaluation")
    parser.add_argument("--output", "-o", default="EVALUATION_RESULTS.md",
                       help="Output file for results (default: EVALUATION_RESULTS.md)")
    parser.add_argument("--model", choices=["REBEL", "SpERT", "all"], default="all",
                       help="Which model to evaluate (default: all)")
    parser.add_argument("--experiment", choices=["FG", "CG+FG", "all"], default="all",
                       help="Which experiment type to evaluate (default: all)")

    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator()

    if args.model == "all" and args.experiment == "all":
        # Run complete evaluation
        evaluator.run_complete_evaluation(args.output)
    else:
        # Run specific evaluation
        print(f"Running evaluation for {args.model} {args.experiment}")
        # Implement specific evaluation logic here

if __name__ == "__main__":
    main()

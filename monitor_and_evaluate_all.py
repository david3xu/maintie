#!/usr/bin/env python3
"""
🏆 COMPREHENSIVE MONITORING AND EVALUATION SCRIPT
Monitors all REBEL and SpERT training progress and runs final evaluation

This script:
1. Monitors all running training processes
2. Tracks completion of experiments
3. Automatically runs comprehensive evaluation when models are ready
4. Generates final RESULTS.md style output
"""

import os
import sys
import time
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

class ExperimentMonitor:
    """Monitors all MaintIE experiments and runs evaluation when ready"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.rebel_experiments_dir = self.base_dir / "experiments"
        self.spert_models_dir = self.base_dir / "models" / "spert" / "data" / "save"

        # Expected experiments
        self.rebel_experiments = {
            "FG-0": "rebel_fg_0",
            "FG-1": "rebel_final_corrected",  # Already completed
            "FG-2": "rebel_fg_2",
            "FG-3": "rebel_fg_3",
            "CG+FG-0": "rebel_cgfg_0",
            "CG+FG-1": "rebel_cgfg_1",
            "CG+FG-2": "rebel_cgfg_2",
            "CG+FG-3": "rebel_cgfg_3",
        }

        self.spert_experiments = {
            "FG-0": "maintie_g_0_train",
            "FG-1": "maintie_g_1_train",
            "FG-2": "maintie_g_2_train",
            "FG-3": "maintie_g_3_train",
            "CG+FG-0": "maintie_gs_0_train",
            "CG+FG-1": "maintie_gs_1_train",
            "CG+FG-2": "maintie_gs_2_train",
            "CG+FG-3": "maintie_gs_3_train",
        }

        # Base models needed for sequential fine-tuning
        self.spert_base_models = {
            "Silver-0": "maintie_s_0_train",
            "Silver-1": "maintie_s_1_train",
            "Silver-2": "maintie_s_2_train",
            "Silver-3": "maintie_s_3_train",
        }

        self.rebel_base_model = "maintie_base_model"  # For REBEL CG+FG experiments

    def check_training_processes(self) -> Dict[str, Any]:
        """Check what training processes are currently running"""
        running_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 1:
                    cmd_str = ' '.join(cmdline)

                    # Check for REBEL training
                    if 'train.py' in cmd_str and 'rebel' in cmd_str:
                        running_processes.append({
                            'type': 'REBEL',
                            'pid': proc.info['pid'],
                            'command': cmd_str,
                            'name': self._extract_experiment_name(cmd_str)
                        })

                    # Check for SpERT training
                    elif 'spert.py' in cmd_str and 'train' in cmd_str:
                        running_processes.append({
                            'type': 'SpERT',
                            'pid': proc.info['pid'],
                            'command': cmd_str,
                            'name': self._extract_spert_experiment_name(cmd_str)
                        })

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            'running_processes': running_processes,
            'total_running': len(running_processes)
        }

    def _extract_experiment_name(self, cmd_str: str) -> str:
        """Extract experiment name from REBEL command"""
        if 'maintie_g_0' in cmd_str:
            return 'FG-0'
        elif 'maintie_g_1' in cmd_str:
            return 'FG-1'
        elif 'maintie_g_2' in cmd_str:
            return 'FG-2'
        elif 'maintie_g_3' in cmd_str:
            return 'FG-3'
        elif 'maintie_s_1' in cmd_str:
            return 'Base-Model'
        else:
            return 'Unknown'

    def _extract_spert_experiment_name(self, cmd_str: str) -> str:
        """Extract experiment name from SpERT command"""
        if 'maintie_g_0' in cmd_str:
            return 'FG-0'
        elif 'maintie_g_1' in cmd_str:
            return 'FG-1'
        elif 'maintie_g_2' in cmd_str:
            return 'FG-2'
        elif 'maintie_g_3' in cmd_str:
            return 'FG-3'
        elif 'maintie_gs_0' in cmd_str:
            return 'CG+FG-0'
        elif 'maintie_gs_1' in cmd_str:
            return 'CG+FG-1'
        elif 'maintie_gs_2' in cmd_str:
            return 'CG+FG-2'
        elif 'maintie_gs_3' in cmd_str:
            return 'CG+FG-3'
        elif 'maintie_s_' in cmd_str:
            return 'Silver-Base'
        else:
            return 'Unknown'

    def check_completed_experiments(self) -> Dict[str, Any]:
        """Check which experiments have completed successfully"""
        completed = {
            'REBEL': {},
            'SpERT': {},
            'SpERT_Base': {}
        }

        # Check REBEL experiments
        for exp_name, exp_dir in self.rebel_experiments.items():
            exp_path = self.rebel_experiments_dir / exp_dir
            if exp_path.exists():
                ckpt_files = list(exp_path.glob("*.ckpt"))
                if ckpt_files:
                    completed['REBEL'][exp_name] = {
                        'path': str(exp_path),
                        'checkpoints': [str(f) for f in ckpt_files],
                        'latest_checkpoint': str(max(ckpt_files, key=lambda x: x.stat().st_mtime))
                    }

        # Check SpERT base models
        for exp_name, model_name in self.spert_base_models.items():
            model_path = self.spert_models_dir / model_name
            if model_path.exists():
                completed['SpERT_Base'][exp_name] = {
                    'path': str(model_path),
                    'model_name': model_name
                }

        # Check SpERT experiments
        for exp_name, model_name in self.spert_experiments.items():
            model_path = self.spert_models_dir / model_name
            if model_path.exists():
                completed['SpERT'][exp_name] = {
                    'path': str(model_path),
                    'model_name': model_name
                }

        return completed

    def check_requirements_for_sequential_training(self, completed: Dict[str, Any]) -> Dict[str, bool]:
        """Check if base models are ready for sequential fine-tuning"""
        requirements = {
            'REBEL_CG+FG': False,
            'SpERT_CG+FG': True  # Assume SpERT base models are ready
        }

        # Check if REBEL base model exists (for CG+FG experiments)
        rebel_base_path = self.rebel_experiments_dir / self.rebel_base_model
        if rebel_base_path.exists():
            ckpt_files = list(rebel_base_path.glob("*.ckpt"))
            requirements['REBEL_CG+FG'] = len(ckpt_files) > 0

        # Check SpERT base models
        spert_base_ready = all(
            exp_name in completed['SpERT_Base']
            for exp_name in self.spert_base_models.keys()
        )
        requirements['SpERT_CG+FG'] = spert_base_ready

        return requirements

    def print_status_report(self):
        """Print a comprehensive status report"""
        print(f"\n{'='*80}")
        print(f"🏆 MAINTIE EXPERIMENT STATUS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

        # Check running processes
        processes = self.check_training_processes()
        print(f"\n🚀 RUNNING TRAINING PROCESSES ({processes['total_running']} active)")
        print("-" * 40)

        if processes['running_processes']:
            for proc in processes['running_processes']:
                print(f"   ✅ {proc['type']} {proc['name']} (PID: {proc['pid']})")
        else:
            print("   No training processes currently running")

        # Check completed experiments
        completed = self.check_completed_experiments()

        print(f"\n🤖 REBEL EXPERIMENTS")
        print("-" * 40)
        for exp_name in self.rebel_experiments.keys():
            if exp_name in completed['REBEL']:
                print(f"   ✅ {exp_name} - COMPLETED")
            else:
                print(f"   ⏳ {exp_name} - PENDING")

        print(f"\n🎯 SPERT BASE MODELS")
        print("-" * 40)
        for exp_name in self.spert_base_models.keys():
            if exp_name in completed['SpERT_Base']:
                print(f"   ✅ {exp_name} - COMPLETED")
            else:
                print(f"   ⏳ {exp_name} - PENDING")

        print(f"\n🎯 SPERT EXPERIMENTS")
        print("-" * 40)
        for exp_name in self.spert_experiments.keys():
            if exp_name in completed['SpERT']:
                print(f"   ✅ {exp_name} - COMPLETED")
            else:
                print(f"   ⏳ {exp_name} - PENDING")

        # Check sequential training readiness
        requirements = self.check_requirements_for_sequential_training(completed)
        print(f"\n🔄 SEQUENTIAL FINE-TUNING READINESS")
        print("-" * 40)
        for req_name, ready in requirements.items():
            status = "✅ READY" if ready else "⏳ WAITING"
            print(f"   {req_name}: {status}")

        # Calculate completion percentage
        total_rebel = len(self.rebel_experiments)
        completed_rebel = len(completed['REBEL'])
        total_spert = len(self.spert_experiments)
        completed_spert = len(completed['SpERT'])

        rebel_progress = (completed_rebel / total_rebel) * 100
        spert_progress = (completed_spert / total_spert) * 100
        overall_progress = ((completed_rebel + completed_spert) / (total_rebel + total_spert)) * 100

        print(f"\n📊 PROGRESS SUMMARY")
        print("-" * 40)
        print(f"   REBEL Progress: {completed_rebel}/{total_rebel} ({rebel_progress:.1f}%)")
        print(f"   SpERT Progress: {completed_spert}/{total_spert} ({spert_progress:.1f}%)")
        print(f"   Overall Progress: {overall_progress:.1f}%")

        return completed, requirements, overall_progress

    def can_run_comprehensive_evaluation(self, completed: Dict[str, Any]) -> bool:
        """Check if we can run comprehensive evaluation"""
        # We can run evaluation if we have at least some completed experiments
        return len(completed['REBEL']) > 0 or len(completed['SpERT']) > 0

    def run_comprehensive_evaluation(self):
        """Run the comprehensive evaluation script"""
        print(f"\n🏆 RUNNING COMPREHENSIVE EVALUATION")
        print("=" * 50)

        try:
            cmd = ["python", "comprehensive_evaluation.py", "--output", "FINAL_RESULTS.md"]
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )

            print("✅ Comprehensive evaluation completed!")
            print(f"📄 Results saved to: FINAL_RESULTS.md")

            if result.stdout:
                print(f"\nEvaluation output:\n{result.stdout}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            return False

    def monitor_continuously(self, check_interval: int = 300):
        """Continuously monitor experiments and run evaluation when ready"""
        print("🔄 Starting continuous monitoring...")
        print(f"Checking every {check_interval} seconds")
        print("Press Ctrl+C to stop monitoring")

        last_evaluation_run = False

        try:
            while True:
                completed, requirements, progress = self.print_status_report()

                # Run evaluation if we have completed experiments and haven't run it yet
                if not last_evaluation_run and self.can_run_comprehensive_evaluation(completed):
                    if progress >= 25:  # Run when at least 25% complete
                        print(f"\n🎯 Sufficient progress ({progress:.1f}%) - Running evaluation...")
                        if self.run_comprehensive_evaluation():
                            last_evaluation_run = True

                # Check if everything is complete
                if progress >= 100:
                    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
                    if not last_evaluation_run:
                        self.run_comprehensive_evaluation()
                    break

                print(f"\n⏰ Next check in {check_interval} seconds...")
                time.sleep(check_interval)

        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")

            if not last_evaluation_run:
                completed, _, _ = self.check_completed_experiments(), None, None
                if self.can_run_comprehensive_evaluation(completed):
                    print("🎯 Running final evaluation before exit...")
                    self.run_comprehensive_evaluation()

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Monitor MaintIE experiments and run evaluation")
    parser.add_argument("--monitor", action="store_true",
                        help="Run continuous monitoring")
    parser.add_argument("--status", action="store_true",
                        help="Show current status and exit")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run comprehensive evaluation now")
    parser.add_argument("--interval", type=int, default=300,
                        help="Monitoring check interval in seconds (default: 300)")

    args = parser.parse_args()

    monitor = ExperimentMonitor()

    if args.status:
        monitor.print_status_report()
    elif args.evaluate:
        monitor.run_comprehensive_evaluation()
    elif args.monitor:
        monitor.monitor_continuously(args.interval)
    else:
        # Default: show status and ask what to do
        completed, requirements, progress = monitor.print_status_report()

        if progress < 100:
            print(f"\n💡 Options:")
            print(f"   --monitor     Start continuous monitoring")
            print(f"   --evaluate    Run evaluation with current completed experiments")
            print(f"   --status      Show this status report")
        else:
            print(f"\n🎉 All experiments complete! Running final evaluation...")
            monitor.run_comprehensive_evaluation()

if __name__ == "__main__":
    main()

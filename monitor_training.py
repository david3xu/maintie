#!/usr/bin/env python3
"""
📊 TRAINING PROGRESS MONITOR
Track full REBEL training progress towards paper performance (F1=67-71%)
"""

import os
import time
import subprocess
from datetime import datetime
import glob

def check_training_status():
    """Check if training process is still running"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )

        for line in result.stdout.split('\n'):
            if 'run_full_training.py' in line and 'python' in line:
                # Extract CPU and memory usage
                parts = line.split()
                if len(parts) >= 11:
                    cpu_usage = parts[2]
                    mem_usage = parts[3]
                    return True, cpu_usage, mem_usage

        return False, "0", "0"

    except Exception as e:
        return False, "0", "0"

def parse_training_log():
    """Parse training log for metrics and progress"""
    log_file = "training_full.log"

    if not os.path.exists(log_file):
        return None

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        status = {
            'total_lines': len(lines),
            'latest_lines': lines[-10:] if len(lines) >= 10 else lines,
            'epochs_completed': 0,
            'current_step': 0,
            'best_f1': 0.0,
            'current_loss': 0.0,
            'errors': [],
            'milestones': []
        }

        # Parse log for key metrics
        for line in lines:
            line = line.strip()

            # Check for epoch completion
            if 'Epoch' in line and 'completed' in line:
                try:
                    epoch_num = int(line.split('Epoch')[1].split()[0])
                    status['epochs_completed'] = max(status['epochs_completed'], epoch_num)
                except:
                    pass

            # Check for step progress
            if 'step' in line.lower() and '/' in line:
                try:
                    if 'step' in line:
                        parts = line.split('step')
                        if len(parts) > 1:
                            step_part = parts[1].split()[0]
                            if step_part.isdigit():
                                status['current_step'] = int(step_part)
                except:
                    pass

            # Check for F1 scores
            if 'val_F1_micro' in line or 'F1' in line:
                try:
                    # Extract F1 score
                    if ':' in line:
                        f1_part = line.split(':')[-1].strip()
                        if '%' in f1_part:
                            f1_val = float(f1_part.replace('%', ''))
                            status['best_f1'] = max(status['best_f1'], f1_val)
                        elif f1_part.replace('.', '').isdigit():
                            f1_val = float(f1_part)
                            if f1_val <= 1.0:  # Assume it's in 0-1 range
                                f1_val *= 100
                            status['best_f1'] = max(status['best_f1'], f1_val)
                except:
                    pass

            # Check for loss
            if 'loss' in line.lower() and ':' in line:
                try:
                    loss_part = line.split(':')[-1].strip()
                    if loss_part.replace('.', '').replace('-', '').isdigit():
                        status['current_loss'] = float(loss_part)
                except:
                    pass

            # Check for errors
            if 'error' in line.lower() or 'failed' in line.lower():
                status['errors'].append(line)

            # Check for milestones
            if any(keyword in line.lower() for keyword in ['training', 'validation', 'checkpoint', 'saved']):
                if len(status['milestones']) < 10:  # Keep last 10 milestones
                    status['milestones'].append(line)

        return status

    except Exception as e:
        return None

def check_lightning_logs():
    """Check PyTorch Lightning logs for detailed metrics"""
    lightning_dir = "lightning_logs"

    if not os.path.exists(lightning_dir):
        return None

    # Find latest version directory
    version_dirs = glob.glob(os.path.join(lightning_dir, "version_*"))
    if not version_dirs:
        return None

    latest_version = max(version_dirs, key=lambda x: int(x.split('_')[-1]))

    metrics = {
        'version': latest_version.split('/')[-1],
        'checkpoints': 0,
        'metrics_file': None
    }

    # Count checkpoints
    checkpoint_dir = os.path.join(latest_version, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        metrics['checkpoints'] = len(checkpoints)

    # Check for metrics file
    metrics_file = os.path.join(latest_version, "metrics.csv")
    if os.path.exists(metrics_file):
        metrics['metrics_file'] = metrics_file
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                metrics['metrics_lines'] = len(lines)
        except:
            metrics['metrics_lines'] = 0

    return metrics

def display_progress():
    """Display comprehensive training progress"""
    print("📊 REBEL FULL TRAINING PROGRESS MONITOR")
    print("=" * 60)
    print(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # Check training status
    is_running, cpu_usage, mem_usage = check_training_status()

    if is_running:
        print("✅ TRAINING STATUS: ACTIVE")
        print(f"   💻 CPU Usage: {cpu_usage}%")
        print(f"   🧠 Memory Usage: {mem_usage}%")
    else:
        print("⚠️ TRAINING STATUS: NOT DETECTED")
        print("   (May have completed or stopped)")
    print("")

    # Parse training log
    log_status = parse_training_log()

    if log_status:
        print("📋 TRAINING PROGRESS:")
        print(f"   📖 Log lines: {log_status['total_lines']}")
        print(f"   🔄 Epochs completed: {log_status['epochs_completed']}/10")
        print(f"   📈 Current step: {log_status['current_step']}/2000")
        print(f"   🎯 Best F1 score: {log_status['best_f1']:.2f}%")
        print(f"   📉 Current loss: {log_status['current_loss']:.4f}")

        # Progress percentage
        step_progress = (log_status['current_step'] / 2000) * 100 if log_status['current_step'] > 0 else 0
        epoch_progress = (log_status['epochs_completed'] / 10) * 100
        print(f"   📊 Step progress: {step_progress:.1f}%")
        print(f"   📊 Epoch progress: {epoch_progress:.1f}%")
        print("")

        # Performance comparison
        print("🎯 PERFORMANCE VS PAPER TARGETS:")
        paper_strict_f1 = 67.87
        paper_loose_f1 = 71.14

        if log_status['best_f1'] > 0:
            strict_diff = log_status['best_f1'] - paper_strict_f1
            loose_diff = log_status['best_f1'] - paper_loose_f1
            print(f"   📊 Current F1: {log_status['best_f1']:.2f}%")
            print(f"   📋 Paper Strict: {paper_strict_f1:.2f}% (diff: {strict_diff:+.2f}%)")
            print(f"   📋 Paper Loose: {paper_loose_f1:.2f}% (diff: {loose_diff:+.2f}%)")

            if log_status['best_f1'] >= paper_strict_f1:
                print("   🎉 REACHED PAPER PERFORMANCE (Strict)!")
            elif log_status['best_f1'] >= paper_strict_f1 * 0.8:
                print("   🔥 APPROACHING PAPER PERFORMANCE!")
            else:
                print("   📚 LEARNING - PROGRESS EXPECTED")
        else:
            print("   📚 F1 metrics not available yet")
        print("")

        # Show latest log lines
        if log_status['latest_lines']:
            print("📝 LATEST LOG OUTPUT:")
            for line in log_status['latest_lines'][-5:]:  # Show last 5 lines
                print(f"   {line.strip()}")
            print("")

        # Show errors if any
        if log_status['errors']:
            print("⚠️ RECENT ERRORS:")
            for error in log_status['errors'][-3:]:  # Show last 3 errors
                print(f"   ❌ {error.strip()}")
            print("")

    # Check Lightning logs
    lightning_metrics = check_lightning_logs()

    if lightning_metrics:
        print("⚡ PYTORCH LIGHTNING METRICS:")
        print(f"   📁 Experiment: {lightning_metrics['version']}")
        print(f"   💾 Checkpoints: {lightning_metrics['checkpoints']}")
        if lightning_metrics['metrics_file']:
            print(f"   📊 Metrics logged: {lightning_metrics.get('metrics_lines', 0)} entries")
        print("")

    # Training time estimation
    if log_status and log_status['current_step'] > 0:
        steps_remaining = 2000 - log_status['current_step']
        if steps_remaining > 0:
            # Rough time estimation (very approximate)
            estimated_hours = (steps_remaining / 2000) * 10  # Assume 10 hours total
            print(f"⏱️ ESTIMATED TIME REMAINING: ~{estimated_hours:.1f} hours")
        else:
            print("⏱️ TRAINING SHOULD BE COMPLETE!")
        print("")

    print("🎯 TARGET MILESTONES:")
    print("   📊 F1 > 30%: Learning mining domain")
    print("   📊 F1 > 50%: Good progress")
    print("   📊 F1 > 60%: Approaching paper")
    print("   📊 F1 > 67%: Paper performance!")
    print("")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor REBEL training progress")
    parser.add_argument("--continuous", "-c", action="store_true",
                       help="Monitor continuously (refresh every 30 seconds)")
    parser.add_argument("--interval", "-i", type=int, default=30,
                       help="Refresh interval in seconds (default: 30)")

    args = parser.parse_args()

    if args.continuous:
        print("🔄 CONTINUOUS MONITORING MODE")
        print("Press Ctrl+C to stop")
        print("")

        try:
            while True:
                display_progress()
                print(f"🔄 Refreshing in {args.interval} seconds...")
                print("=" * 60)
                time.sleep(args.interval)
                # Clear screen
                os.system('clear' if os.name == 'posix' else 'cls')
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
    else:
        display_progress()

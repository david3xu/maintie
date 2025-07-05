# optimize_for_cpu.py
import yaml
import configparser
import os

def create_cpu_optimized_configs():
    """Create CPU-optimized training configurations"""
    
    # SpERT CPU optimizations
    spert_cpu_params = {
        'train_batch_size': '1',
        'eval_batch_size': '2', 
        'epochs': '10',
        'lr': '3e-5',
        'neg_entity_count': '50',
        'neg_relation_count': '50',
        'sampling_processes': '1',
        'max_pairs': '500'
    }
    
    # REBEL CPU optimizations  
    rebel_cpu_params = {
        'train_batch_size': 1,
        'eval_batch_size': 2,
        'gradient_acc_steps': 8,
        'max_steps': 1000,
        'num_train_epochs': 3.0,
        'learning_rate': 0.00003,
        'gpus': 0,  # CPU only
        'precision': 32
    }
    
    # Update REBEL training configs
    rebel_train_configs = [
        'models/rebel/conf/train/maintie_g_0_train.yaml',
        'models/rebel/conf/train/maintie_g_1_train.yaml',
        'models/rebel/conf/train/maintie_g_2_train.yaml',
        'models/rebel/conf/train/maintie_g_3_train.yaml'
    ]
    
    for config_path in rebel_train_configs:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply CPU optimizations
            config.update(rebel_cpu_params)
            
            # Save CPU-optimized version
            cpu_config_path = config_path.replace('.yaml', '_cpu.yaml')
            with open(cpu_config_path, 'w') as f:
                yaml.dump(config, f)
            
            print(f"Created CPU config: {cpu_config_path}")

if __name__ == "__main__":
    create_cpu_optimized_configs()

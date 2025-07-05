# update_rebel_configs.py
import os
import yaml
from pathlib import Path

def update_rebel_config_paths():
    """Update REBEL configuration paths for Azure ML"""
    base_path = Path.cwd()
    
    # Update data configuration paths
    data_configs = [
        'models/rebel/conf/data/maintie_g_0_data.yaml',
        'models/rebel/conf/data/maintie_g_1_data.yaml', 
        'models/rebel/conf/data/maintie_g_2_data.yaml',
        'models/rebel/conf/data/maintie_g_3_data.yaml',
        'models/rebel/conf/data/maintie_s_1_data.yaml'
    ]
    
    for config_path in data_configs:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update paths to current working directory
            if 'train_file' in config:
                config['train_file'] = str(base_path / '/'.join(config['train_file'].split('/')[-3:]))
            if 'dev_file' in config:
                config['dev_file'] = str(base_path / '/'.join(config['dev_file'].split('/')[-3:]))
            if 'test_file' in config:
                config['test_file'] = str(base_path / '/'.join(config['test_file'].split('/')[-3:]))
            
            # Save updated configuration
            azure_config_path = config_path.replace('.yaml', '_azureml.yaml')
            with open(azure_config_path, 'w') as f:
                yaml.dump(config, f)
            
            print(f"Updated: {azure_config_path}")

if __name__ == "__main__":
    update_rebel_config_paths()

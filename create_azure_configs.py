# create_azure_configs.py
import os
import configparser

def update_spert_config(input_file, output_file):
    """Update SpERT config for Azure ML compatibility"""
    config = configparser.ConfigParser()
    config.read(input_file)
    
    # Update paths to be relative
    for section in config.sections():
        if 'train_path' in config[section]:
            config[section]['train_path'] = config[section]['train_path'].replace(
                'D:/Repos/nlp_tlp/maintie/', './'
            )
        if 'valid_path' in config[section]:
            config[section]['valid_path'] = config[section]['valid_path'].replace(
                'D:/Repos/nlp_tlp/maintie/', './'
            )
        if 'types_path' in config[section]:
            config[section]['types_path'] = config[section]['types_path'].replace(
                'D:/Repos/nlp_tlp/maintie/', './'
            )
        
        # Update output paths for Azure ML
        config[section]['log_path'] = './outputs/logs/'
        config[section]['save_path'] = './outputs/models/'
        
        # CPU-optimized parameters
        config[section]['train_batch_size'] = '1'
        config[section]['eval_batch_size'] = '2'
        config[section]['epochs'] = '10'
        
    with open(output_file, 'w') as f:
        config.write(f)

# Convert all configuration files
configs = [
    'maintie_s_0_train.conf',
    'maintie_s_1_train.conf', 
    'maintie_s_2_train.conf',
    'maintie_s_3_train.conf',
    'maintie_g_0_train.conf',
    'maintie_g_1_train.conf',
    'maintie_g_2_train.conf', 
    'maintie_g_3_train.conf'
]

os.makedirs('models/spert/configs/azure-ml', exist_ok=True)

for config_file in configs:
    input_path = f'models/spert/configs/{config_file}'
    output_path = f'models/spert/configs/azure-ml/{config_file.replace(".conf", "_azureml.conf")}'
    
    if os.path.exists(input_path):
        update_spert_config(input_path, output_path)
        print(f"Updated: {output_path}")

# fix_pytorch_lightning.py
import os
import site
import sys

def fix_pytorch_lightning_bug():
    """Fix the documented PyTorch Lightning checkpoint bug"""
    
    # Find pytorch_lightning installation
    for path in site.getsitepackages():
        pl_saving_path = os.path.join(path, 'pytorch_lightning', 'core', 'saving.py')
        
        if os.path.exists(pl_saving_path):
            print(f"Found pytorch_lightning at: {pl_saving_path}")
            
            # Read the file
            with open(pl_saving_path, 'r') as f:
                content = f.read()
            
            # Check if fix already applied
            if 'MaintIE_FIX_APPLIED' in content:
                print("Fix already applied")
                return
            
            # Apply the fix - comment out problematic line
            fixed_content = content.replace(
                'checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)',
                '# checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)  # MaintIE_FIX_APPLIED'
            )
            
            # Write back the fixed file
            with open(pl_saving_path, 'w') as f:
                f.write(fixed_content)
            
            print("PyTorch Lightning bug fix applied successfully")
            return
    
    print("Warning: pytorch_lightning installation not found")

if __name__ == "__main__":
    fix_pytorch_lightning_bug()

import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import datraclass, field

class Config:
    """Configuration manager for the project"""
    def __init__(self, config_path:str):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML config file.
        """

        self.config_path = Path(config_path)
        self.config = self._load_config()

    
    def _load_config(self):
        """Load YAML configuration file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config
    
    def get(self, key:str, default:Any=None):
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key (eg., 'model.person_lstm.hidden_dim)
            default: Default value if key is not found.
        
        Returns:
            Configuration value.
        
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
            
    def __getitem__(self, key:str):
        return self.get(key)
    
    def update(self, updates):
        self._update_recursive(self.config, updates)

    def _update_recursive(self, config:Dict, updates:Dict):
        """Recursively update nested dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in config:
                self._update_recursive(config[key], value)
            else:
                config[key] = value

    def save(self, output_path:str):
        """Save current configuration to YAML file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)



if __name__ == "__main__":
    config = Config('config.yaml')

    print(config.get('model.person_lstm.hidden_dim'))
    print(config['training.batch_size'])

    config.updates({"training": {"learning_rate": 0.0001}})

    config.save('outputs/experiment_config.yaml')


    
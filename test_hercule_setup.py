#!/usr/bin/env python3
"""Test script to display Hercule configuration."""

import logging
from pathlib import Path

from src.hercule.config import load_config_from_yaml
from src.hercule.run import EnvironmentManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def display_config():
    """Load and display the configuration from config_example.yaml."""
    config_path = Path("config_example.yaml")
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    try:
        # Load configuration using the config module function
        config = load_config_from_yaml(config_path)
        logger.info("✓ Configuration loaded successfully from config_example.yaml")
        
        # Display general settings
        print("\n" + "="*60)
        print("HERCULE CONFIGURATION")
        print("="*60)
        
        print("\nGeneral Settings:")
        print(f"  Max iterations: {config.max_iterations}")
        print(f"  Output directory: {config.output_dir}")
        
        # Display environments using config methods and environment manager
        env_names = config.get_environment_names()
        env_configs = config.get_environment_configs()
        print(f"\nEnvironments ({len(env_names)} configured):")
        
        # Use EnvironmentManager to get detailed environment info
        with EnvironmentManager(config) as env_manager:
            for i, env_config in enumerate(env_configs, 1):
                print(f"  {i}. {env_config.name}")
                
                # Get environment info using the run module function
                try:
                    env_info = env_manager.get_environment_info(env_config.name)
                    print("     Environment Info:")
                    print(f"       - Observation space: {env_info['observation_space']['type']}")
                    if env_info['observation_space'].get('shape'):
                        print(f"         Shape: {env_info['observation_space']['shape']}")
                    if env_info['observation_space'].get('low') is not None:
                        print(f"         Low: {env_info['observation_space']['low']}")
                    if env_info['observation_space'].get('high') is not None:
                        print(f"         High: {env_info['observation_space']['high']}")
                    
                    print(f"       - Action space: {env_info['action_space']['type']}")
                    if env_info['action_space'].get('n'):
                        print(f"         Actions: {env_info['action_space']['n']}")
                    if env_info['action_space'].get('shape'):
                        print(f"         Shape: {env_info['action_space']['shape']}")
                    if env_info['action_space'].get('low') is not None:
                        print(f"         Low: {env_info['action_space']['low']}")
                    if env_info['action_space'].get('high') is not None:
                        print(f"         High: {env_info['action_space']['high']}")
                    
                    if env_info.get('max_episode_steps'):
                        print(f"       - Max episode steps: {env_info['max_episode_steps']}")
                
                except Exception as e:
                    print(f"     ⚠️  Could not load environment info: {e}")
                
                # Use the config method to get hyperparameters
                hyperparams = config.get_hyperparameters_for_environment(env_config.name)
                if hyperparams:
                    print("     Configuration Hyperparameters:")
                    for key, value in hyperparams.items():
                        value_type = type(value).__name__
                        if isinstance(value, list) and value:
                            element_type = type(value[0]).__name__ if value else 'unknown'
                            print(f"       - {key}: {value} (list of {element_type})")
                        else:
                            print(f"       - {key}: {value} ({value_type})")
                else:
                    print("     No configuration hyperparameters")
                print()  # Empty line for readability
        
        # Display models using config methods
        model_names = config.get_model_names()
        print(f"\nModels ({len(config.models)} configured):")
        
        for i, model in enumerate(config.models, 1):
            print(f"  {i}. {model.name}")
            # Use the config method to get hyperparameters
            hyperparams = config.get_hyperparameters_for_model(model.name)
            if hyperparams:
                print("     Hyperparameters:")
                for key, value in hyperparams.items():
                    value_type = type(value).__name__
                    if isinstance(value, list) and value:
                        element_type = type(value[0]).__name__ if value else 'unknown'
                        print(f"       - {key}: {value} (list of {element_type})")
                    else:
                        print(f"       - {key}: {value} ({value_type})")
            else:
                print("     No hyperparameters")
        
        # Display summary using config methods
        print("\nSummary:")
        print(f"  Total environments: {len(env_names)}")
        print(f"  Environment names: {', '.join(env_names)}")
        print(f"  Total models: {len(model_names)}")
        print(f"  Model names: {', '.join(model_names)}")
        
        print("\n" + "="*60)
        logger.info("✓ Configuration display completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Failed to load or display configuration: {e}")
        raise


def main():
    """Display configuration from config_example.yaml."""
    logger.info("Hercule Configuration Display")
    display_config()


if __name__ == "__main__":
    main()
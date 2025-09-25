# config.py

import configparser
from typing import Dict, Any


def load_config(config_path: str = 'config.ini') -> Dict[str, Any]:
    """
    Loads configuration from the specified INI file.

    Returns:
        A dictionary containing parsed configuration settings.
    """
    config = configparser.ConfigParser()

    # Set defaults for models and files for robustness
    config.read_dict({
        'MODELS': {
            'tiny': 'tiny',
            'base': 'base',
            'small': 'small',
        },
        'TRANSCRIPTION': {
            'device': 'cuda',
            'model_name': 'base',
            'language': 'None',
        },
        'FILES': {
            'video_extensions': '.mp4, .mkv, .avi, .mov, .wmv',
        }
    })

    # Read user config, which overrides defaults
    read_files = config.read(config_path)
    if not read_files:
        print(f"Warning: Config file not found at '{config_path}'. Using default settings.")

    # Resolve model identifier
    selected_model_name = config.get('TRANSCRIPTION', 'model_name', fallback='base')
    model_identifier = config.get('MODELS', selected_model_name, fallback=selected_model_name)

    # Resolve device and language
    device = config.get('TRANSCRIPTION', 'device', fallback='cuda').lower()
    language_str = config.get('TRANSCRIPTION', 'language', fallback='None')
    language = language_str if language_str.lower() != 'none' else None

    return {
        'model_identifier': model_identifier,
        'device': device,
        'language': language,
        'video_extensions': [ext.strip().lower() for ext in config.get('FILES', 'video_extensions').split(',')],
    }
import configparser
from typing import Dict, Any
import multiprocessing


def load_config(config_path: str = 'config.ini') -> Dict[str, Any]:
    """
    Loads configuration from the specified INI file.

    Returns:
        A dictionary containing parsed configuration settings.
    """
    config = configparser.ConfigParser()

    cpu_count = multiprocessing.cpu_count()

    # Set robust defaults including new settings
    config.read_dict({
        'MODELS': {
            'tiny': 'tiny',
            'base': 'base',
            'small': 'small',
            'medium': 'medium',
            'large-turbo': 'large-v3-turbo',
        },
        'TRANSCRIPTION': {
            'device': 'cuda',
            'model_name': 'base',
            'language': 'None',
            'compute_type': 'float16',
            'parallel_workers': str(cpu_count)
        },
        'FILES': {
            'video_extensions': '.mp4, .mkv, .avi, .mov, .wmv',
        }
    })

    # Read user config, which overrides defaults
    read_files = config.read(config_path)
    if not read_files:
        print(f"Warning: Config file not found at '{config_path}'. Using default settings.")

    # Resolve settings
    selected_model_name = config.get('TRANSCRIPTION', 'model_name', fallback='base')
    model_identifier = config.get('MODELS', selected_model_name, fallback=selected_model_name)

    device = config.get('TRANSCRIPTION', 'device', fallback='cuda').lower()
    language_str = config.get('TRANSCRIPTION', 'language', fallback='None')
    language = language_str if language_str.lower() != 'none' else None

    compute_type = config.get('TRANSCRIPTION', 'compute_type', fallback='float16').lower()

    try:
        parallel_workers = config.getint('TRANSCRIPTION', 'parallel_workers', fallback=cpu_count)
    except ValueError:
        print("Warning: Invalid value for parallel_workers. Defaulting to CPU count.")
        parallel_workers = cpu_count

    return {
        'model_identifier': model_identifier,
        'device': device,
        'language': language,
        'compute_type': compute_type,
        'video_extensions': [ext.strip().lower() for ext in config.get('FILES', 'video_extensions').split(',')],
        'parallel_workers': parallel_workers,
    }
# E:\Github_Repo\Info-Retrieve-AI\setup.py
import sys
import os
import logging
import config

def setup_environment():
    project_root = 'E:\\Github_Repo\\Info-Retrieve-AI'
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Configure logging
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'project_log.log')
    logging.basicConfig(filename=log_file_path,
                        filemode='a',
                        format='%(asctime)s, %(name)s [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)

def get_config():
    return config

# Initialize the environment and import config
setup_environment()
cfg = get_config()

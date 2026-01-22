
"""
setup.py

Ensures required project directories exist.
This makes the project robust across machines and environments.
"""

import os

REQUIRED_DIRS = [
    "data",
    "data/raw",
    "data/processed",
    "models",
    "outputs",
    "test_videos",
    "wandb"
]

def setup_project_structure():
    """
    Create required folders if they do not exist.
    Safe to call multiple times.
    """
    for d in REQUIRED_DIRS:
        os.makedirs(d, exist_ok=True)

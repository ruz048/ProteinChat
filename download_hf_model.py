#!/usr/bin/env python3
"""
Script to download a Hugging Face model repository to a local directory.
"""
import os
from huggingface_hub import snapshot_download

def download_hf_model(repo_id, local_dir):
    """
    Download a Hugging Face model repository.

    Args:
        repo_id: The Hugging Face repository ID (e.g., 'lmsys/vicuna-13b-v1.5')
        local_dir: Local directory path to save the model
    """
    print(f"Downloading {repo_id} to {local_dir}...")

    # Create the directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Set cache directory on the same filesystem as the destination
    # This avoids cross-device link errors
    cache_dir = os.path.join(os.path.dirname(local_dir), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Using cache directory: {cache_dir}")

    # Download the model
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    print(f"Download complete! Model saved to {local_dir}")

if __name__ == "__main__":
    # Configuration
    REPO_ID = "biomap-research/proteinglm-1b-mlm"
    LOCAL_DIR = "/data3/ruiyi/proteinglm-1b-mlm"

    download_hf_model(REPO_ID, LOCAL_DIR)

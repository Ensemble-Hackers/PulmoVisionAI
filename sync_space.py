"""
PulmoVision AI - 1-Click Hugging Face Space Synchronizer
Uploads the local backend/ folder directly to your Hugging Face Space from the terminal.
Usage:
    python sync_space.py
    python sync_space.py hf_yourTokenHere
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"


def sync():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if len(sys.argv) > 1 and sys.argv[1].startswith("hf_"):
        token = sys.argv[1]

    space_id = os.environ.get("HF_SPACE_ID", "stellar-merge/pulmovision-api")

    print(f"\n=======================================================")
    print(f" Syncing backend/ -> Hugging Face Space: {space_id}")
    print(f"=======================================================\n")

    api = HfApi(token=token)

    try:
        ignore_patterns = ["__pycache__/*", "*.pyc", ".env", "*.tmp"]

        api.upload_folder(
            folder_path=str(BACKEND_DIR),
            repo_id=space_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="chore: sync backend files to Space via sync_space.py",
        )
        print("\n [SUCCESS] Successfully synced backend files to Hugging Face Space!")
        print(f" Live Space URL: https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        print(f"\n [ERROR] Sync failed: {e}")
        if "401" in str(e) or "unauthorized" in str(e).lower() or "token" in str(e).lower():
            print("\nTip: Hugging Face requires a write access token to push from the terminal.")
            print("1. Get a write token from: https://huggingface.co/settings/tokens")
            print("2. Run: python sync_space.py hf_yourWriteTokenHere\n")


if __name__ == "__main__":
    sync()

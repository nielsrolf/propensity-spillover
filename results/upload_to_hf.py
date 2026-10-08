"""Maintainer script: upload results/runs and results/introspection to the Hugging Face dataset.

    huggingface-cli login
    python results/upload_to_hf.py
"""
import pathlib

from huggingface_hub import HfApi

from download import HF_REPO

HERE = pathlib.Path(__file__).resolve().parent
api = HfApi()
api.create_repo(HF_REPO, repo_type="dataset", exist_ok=True)
api.upload_folder(repo_id=HF_REPO, repo_type="dataset", folder_path=str(HERE),
                  allow_patterns=["runs/*", "introspection/**", "README.md"])
print("uploaded to https://huggingface.co/datasets/" + HF_REPO)

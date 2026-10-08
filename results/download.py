"""Download the per-response data (results/runs/*.csv.gz, results/introspection/) from Hugging Face.

    pip install huggingface_hub
    python results/download.py
"""
import pathlib

HF_REPO = "nielsrolf/propensity-spillover"   # Hugging Face dataset repo
HERE = pathlib.Path(__file__).resolve().parent


def main():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(HERE),
                      allow_patterns=["runs/*", "introspection/**"])
    print("downloaded to", HERE)


if __name__ == "__main__":
    main()

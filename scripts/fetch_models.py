"""Download the model weights once into the Hugging Face cache (about 2 GB), so runs work offline afterwards.

- depth-anything/DA3-BASE (Apache 2.0): camera poses and depth from several images (video and photo tiers)
- depth-anything/DA3METRIC-LARGE (Apache 2.0): depth in metres, sets the scale (video and photo tiers)
- openai/clip-vit-base-patch32 (MIT): zero-shot damage detection (all tiers)
"""
from huggingface_hub import snapshot_download

MODELS = {
    "depth-anything/DA3-BASE": None,
    "depth-anything/DA3METRIC-LARGE": None,
    "openai/clip-vit-base-patch32": ["*.json", "*.txt", "model.safetensors"],
}

if __name__ == "__main__":
    for repo, patterns in MODELS.items():
        path = snapshot_download(repo, allow_patterns=patterns)
        print(f"{repo}: {path}")

#huggingface-cli login
# hf download ILSVRC/imagenet-1k --repo-type dataset

from huggingface_hub import snapshot_download
import glob, os

root = snapshot_download(
    "timm/imagenet-1k-wds",
    repo_type="dataset",
    cache_dir="/data/datasets",
    allow_patterns=["imagenet1k-train-*.tar", "imagenet1k-validation-*.tar"]
)

train_shards = sorted(glob.glob(os.path.join(root, "imagenet1k-train-*.tar")))
val_shards   = sorted(glob.glob(os.path.join(root, "imagenet1k-validation-*.tar")))
print(len(train_shards), "train,", len(val_shards), "val")  # expect 1024 train, 63 val
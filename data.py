"""
Data loader factory for different dataset types.
Supports ImageFolder and WebDataset (via DALI) loaders.
"""

import torch
import numpy as np
import os
import glob
from torch.utils.data import DataLoader

# DALI imports for WebDataset
try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False
    # Define dummy pipeline_def to avoid NameError on decorators
    def pipeline_def(func):
        return func


def parse_ascii_label(label_uint8):
    """Parse ASCII label from DALI byte array"""
    try:
        label_str = label_uint8.tobytes().decode('utf-8')
        return np.array([int(label_str)], dtype=np.int64)
    except Exception:
        # Fallback for safety
        return np.array([-1], dtype=np.int64)


def _resolve_wds_index_paths(config, split):
    """
    Helper for WebDataset loaders that optionally use precomputed .idx files.

    Each split can point to a subdirectory containing idx files via
    `config.data.index_root` + `{split}_index_subdir`. Returning the list of
    idx files allows DALI to skip the expensive tar scanning step.
    """
    index_root = getattr(config.data, 'index_root', None)
    if not index_root:
        return None

    subdir = getattr(config.data, f'{split}_index_subdir', split)
    pattern = os.path.join(index_root, subdir, '*.idx')
    paths = sorted(glob.glob(pattern))
    return paths if paths else None


@pipeline_def
def imagenet_wds_train_pipeline(shards, img_size=224, shard_id=0, num_shards=1):
    """DALI pipeline for WebDataset training - with augmentations"""
    # Read webdataset tar shards
    jpegs = fn.readers.webdataset(
        paths=shards,
        ext=["jpg"],
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True,
        missing_component_behavior="skip",
        name="wds_reader",
        read_ahead=False,
        pad_last_batch=False,
        lazy_init=True,
    )

    # Decode JPEGs with nvJPEG on GPU
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    # Train-time augmentations on GPU
    images = fn.random_resized_crop(images, size=img_size)
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

    # Normalize to [-1, 1]
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    )
    return images


@pipeline_def
def imagenet_wds_val_pipeline(shards, img_size=224, val_resize=256, shard_id=0, num_shards=1):
    """DALI pipeline for WebDataset validation - NO augmentations, deterministic"""
    # Read webdataset tar shards
    jpegs = fn.readers.webdataset(
        paths=shards,
        ext=["jpg"],
        random_shuffle=False,  # Deterministic order for validation
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True,
        missing_component_behavior="skip",
        name="wds_reader",
        read_ahead=False,
        pad_last_batch=False,
        lazy_init=True,
    )

    # Decode JPEGs with nvJPEG on GPU
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    # Validation preprocessing: resize -> center crop (NO random augmentations)
    images = fn.resize(images, resize_shorter=val_resize)
    images = fn.crop(images, crop=(img_size, img_size), crop_pos_x=0.5, crop_pos_y=0.5)

    # Normalize to [-1, 1]
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    )
    return images


def setup_webdataset_loader(config, device, distributed, local_rank, world_size, split='train', deterministic=False, return_labels=False):
    """Setup WebDataset loader using DALI for GPU-direct loading

    Args:
        deterministic: If True, use validation transforms (resize+center crop) even for training.
                      Used for extracting latents where we need reproducible preprocessing.
    """
    if not DALI_AVAILABLE:
        raise ImportError(
            "DALI is required for WebDataset loading but not installed. "
            "Install with: uv sync --extra dali  (or: pip install nvidia-dali-cuda120 --extra-index-url https://pypi.nvidia.com)"
        )

    # Resolve WDS root from env override or config
    env_root = (os.environ.get('IMAGENET_WDS_ROOT') or
                os.environ.get('HF_WDS_ROOT') or
                os.environ.get('HF_IMAGENET_WDS_ROOT'))
    hf_root = env_root if env_root else getattr(config.data, 'hf_root', None)

    if hf_root is None or not os.path.isdir(hf_root):
        raise FileNotFoundError(
            f"ImageNet WDS root not found. Set config.data.hf_root to a valid directory "
            f"or export IMAGENET_WDS_ROOT=/path/to/imagenet-wds. Current value: {hf_root}"
        )

    # Select train or val shards
    if split == 'train':
        shard_glob = getattr(config.data, 'train_glob', 'imagenet1k-train-*.tar')
    else:  # val
        shard_glob = getattr(config.data, 'val_glob', 'imagenet1k-validation-*.tar')

    shards = sorted(glob.glob(os.path.join(hf_root, shard_glob)))

    if len(shards) == 0:
        raise FileNotFoundError(f"No {split} shards matched at {os.path.join(hf_root, shard_glob)}")

    img_size = config.data.img_size
    val_resize = getattr(config.data, 'val_resize', 256)
    batch_size = config.data.batch_size
    num_threads = getattr(config.data, 'num_workers', 4)

    # Create DALI pipeline (train or val)
    # Use deterministic transforms for both train and val if deterministic=True
    
    # If returning labels, use the latent pipeline which returns (image, label)
    if return_labels:
        pipeline_fn = imagenet_wds_latent_val_pipeline if (split != 'train' or deterministic) else imagenet_wds_latent_train_pipeline
        
        # Need to resolve index paths for latent pipeline
        # index_paths = _resolve_wds_index_paths(config, split)
        index_paths = None  # Disable index files to avoid DALI errors
        
        pipe = pipeline_fn(
            shards=shards,
            index_paths=index_paths,
            img_size=img_size,
            val_resize=val_resize,
            shard_id=local_rank if distributed else 0,
            num_shards=world_size if distributed else 1,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank if distributed else 0,
        )
        pipe.build()
        
        loader = DALIGenericIterator(
            pipe,
            ['data', 'label'],
            size=-1,  # Infinite iterator
            auto_reset=True,
            reader_name="wds_latent_reader"
        )
        
        print(f"DALI WebDataset: Loaded {len(shards)} {split} shards from {hf_root}")
        print(f"DALI WebDataset: Device {local_rank}, processing shards {local_rank}/{world_size}")
        if deterministic and split == 'train':
            print(f"DALI WebDataset: Using deterministic transforms for training (Resize+CenterCrop)")
        print(f"DALI WebDataset: Returning labels")
            
        return loader

    if split == 'train' and not deterministic:
        # Normal training with augmentations
        pipe = imagenet_wds_train_pipeline(
            shards=shards,
            img_size=img_size,
            shard_id=local_rank if distributed else 0,
            num_shards=world_size if distributed else 1,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank if distributed else 0,
        )
    else:  # val or deterministic train
        # Use validation pipeline (resize + center crop) for deterministic preprocessing
        pipe = imagenet_wds_val_pipeline(
            shards=shards,
            img_size=img_size,
            val_resize=val_resize,
            shard_id=local_rank if distributed else 0,
            num_shards=world_size if distributed else 1,
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=local_rank if distributed else 0,
        )
    pipe.build()

    # Wrap with PyTorch iterator
    loader = DALIGenericIterator(
        pipe,
        ['data'],
        size=-1,  # Infinite iterator
        auto_reset=True,
        reader_name="wds_reader"
    )

    print(f"DALI WebDataset: Loaded {len(shards)} {split} shards from {hf_root}")
    print(f"DALI WebDataset: Device {local_rank}, processing shards {local_rank}/{world_size}")
    if deterministic and split == 'train':
        print(f"DALI WebDataset: Using deterministic transforms for training (Resize+CenterCrop)")

    return loader


@pipeline_def
def imagenet_wds_latent_train_pipeline(
    shards,
    img_size=224,
    val_resize=256,
    shard_id=0,
    num_shards=1,
    index_paths=None,
):
    """
    ImageNet WDS pipeline tailored for latent extraction.

    Differences vs. the main train pipeline:
      * Reads paired `jpg` + `cls` components so we retain class labels.
      * Uses resize -> random crop + horizontal flip to stay consistent with
        standard ImageNet training, but still deterministic when needed.
      * Outputs normalized tensors in [-1, 1] plus INT64 labels.
    """
    jpegs, labels = fn.readers.webdataset(
        paths=shards,
        index_paths=index_paths,
        ext=['jpg', 'cls'],
        dtypes=[types.UINT8, types.UINT8],
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True,
        missing_component_behavior="skip",
        name="wds_latent_reader",
        read_ahead=False,
        pad_last_batch=False,
        lazy_init=True,
    )

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_shorter=val_resize)
    images = fn.crop(
        images,
        crop=(img_size, img_size),
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
    )
    images = fn.flip(
        images,
        horizontal=fn.random.coin_flip(probability=0.5),
    )
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    )

    # Reshape label to scalar tensor and cast to INT64
    # labels = fn.reshape(labels, shape=[1])
    # labels = fn.cast(labels, dtype=types.INT64)
    
    # Parse ASCII labels using Python function
    labels = fn.python_function(labels, function=parse_ascii_label, num_outputs=1)
    
    return images, labels


@pipeline_def
def imagenet_wds_latent_val_pipeline(
    shards,
    img_size=224,
    val_resize=256,
    shard_id=0,
    num_shards=1,
    index_paths=None,
):
    """Validation counterpart (resize + center crop)."""
    jpegs, labels = fn.readers.webdataset(
        paths=shards,
        index_paths=index_paths,
        ext=['jpg', 'cls'],
        dtypes=[types.UINT8, types.UINT8],
        random_shuffle=False,
        shard_id=shard_id,
        num_shards=num_shards,
        stick_to_shard=True,
        missing_component_behavior="skip",
        name="wds_latent_reader",
        read_ahead=False,
        pad_last_batch=False,
        lazy_init=True,
    )

    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.resize(images, resize_shorter=val_resize)
    images = fn.crop(
        images,
        crop=(img_size, img_size),
        crop_pos_x=0.5,
        crop_pos_y=0.5,
    )
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    )

    # labels = fn.reshape(labels, shape=[1])
    # labels = fn.cast(labels, dtype=types.INT64)
    
    # Parse ASCII labels using Python function
    labels = fn.python_function(labels, function=parse_ascii_label, num_outputs=1)
    
    return images, labels



def setup_imagefolder_loader(config, device, distributed, local_rank, world_size, split='train', deterministic=False, return_labels=False):
    """Setup simple ImageFolder loader for any directory structure

    Args:
        deterministic: If True, use validation transforms (resize+center crop) even for training.
                      Used for extracting latents where we need reproducible preprocessing.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, DistributedSampler

    img_size = config.data.img_size
    val_resize = getattr(config.data, 'val_resize', 256)

    # Get data directory
    data_dir = getattr(config.data, 'data_dir', None)
    if data_dir is None:
        raise ValueError("config.data.data_dir must be set for ImageFolder loader")

    # Build transforms
    if split == 'train' and not deterministic:
        # Normal training transforms: random crop + flip
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
    else:  # val or deterministic train
        # Validation or deterministic transforms: resize + center crop
        transform = transforms.Compose([
            transforms.Resize(val_resize, antialias=True),  # Add antialias for better quality
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])

    # Create dataset
    if split == 'train':
        train_dir = os.path.join(data_dir, 'train')
    else:
        val_dir = os.path.join(data_dir, 'val')
    dataset = datasets.ImageFolder(train_dir if split == 'train' else val_dir, transform=transform)

    # Create sampler for distributed training
    sampler = None
    shuffle = (split == 'train')
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(config.data, 'num_workers', 4),
        pin_memory=True,
        drop_last=(split == 'train')
    )

    print(f"ImageFolder: Loaded {len(dataset)} images from {data_dir}")

    return loader


def get_data_loader(config, device, distributed=False, local_rank=0, world_size=1, split='train', deterministic=False, return_labels=False):
    """
    Factory function to get data loader based on config.

    Args:
        config: OmegaConf configuration
        device: torch device
        distributed: bool, whether using DDP
        local_rank: local GPU rank
        world_size: total number of GPUs
        split: 'train' or 'val'
        deterministic: If True, use validation transforms (resize+center crop) even for training.
                      Used for extracting latents where we need reproducible preprocessing.

    Returns:
        DataLoader instance
    """
    dataset_type = config.data.dataset.lower()

    if dataset_type in ["imagenet_wds", "webdataset", "wds"]:
        return setup_webdataset_loader(config, device, distributed, local_rank, world_size, split=split, deterministic=deterministic, return_labels=return_labels)
    elif dataset_type == "imagefolder":
        return setup_imagefolder_loader(config, device, distributed, local_rank, world_size, split=split, deterministic=deterministic, return_labels=return_labels)
    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Supported: 'imagefolder', 'imagenet_wds'"
        )

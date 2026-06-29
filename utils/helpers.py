import os
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from torchvision import transforms
import logging


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        import albumentations
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def create_image_folder_dataloader(args,val=False):
    dataset = ImagePaths(args.dataset_path, size=args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
        pin_memory=True,
        drop_last=True,
        shuffle = not val
    )
    return dataloader

def create_webdataset_transform(size=256, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def load_webdataset_for_lightning(args,val=False):
    import webdataset as wds
    from webdataset.shardlists import split_by_node
    train_transform = create_webdataset_transform(size=args.image_size, is_train=True)
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    # Shard splitting and WebDataset creation
    urls = wds.shardlists.split_by_worker(wds.shardlists.expand_urls(args.dataset_path))

    if val:
        dataset = (
            wds.WebDataset(urls, handler=wds.handlers.warn_and_continue, empty_check=False,nodesplitter=split_by_node)
            .decode("pil")
            .rename(image="jpg;jpeg;png")
            .map_dict(image=train_transform)
            .to_tuple("image")
            .with_length(args.dataset_size)
        )
    else:
        dataset = (
            wds.WebDataset(urls, handler=wds.handlers.warn_and_continue, empty_check=False,nodesplitter=split_by_node)
            .shuffle(1000)
            .decode("pil")
            .rename(image="jpg;jpeg;png")
            .map_dict(image=train_transform)
            .to_tuple("image")
            .with_length(args.dataset_size)
        )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    return data_loader

    


def load_data(args):
    """Load data using WebDataset or fallback to ImagePaths."""
    import webdataset as wds
    from webdataset.shardlists import split_by_node
    logger = logging.getLogger("vqgan_training")
    
    # Check if distributed training is enabled
    is_distributed = dist.is_initialized()
    world_size = dist.get_world_size() if is_distributed else 1
    rank = dist.get_rank() if is_distributed else 0
    
    # Check if we should use WebDataset
    use_webdataset = args.dataset_path.endswith('.tar') or '{' in args.dataset_path
    
    if use_webdataset:
        logger.info(f"Using WebDataset with path pattern: {args.dataset_path}")
        train_transform = create_webdataset_transform(size=args.image_size if hasattr(args, 'image_size') else 256)
        
        # For Lightning with WebDataset, we need to handle single-node multi-GPU differently
        if is_distributed:
            logger.info(f"Rank {rank}: Setting up WebDataset for distributed training (world_size={world_size})")
            
            # IMPORTANT: For single-node multi-GPU, we don't need a nodesplitter
            # We just need to ensure each GPU gets different data via the DataLoader
            dataset = (
                wds.WebDataset(args.dataset_path, handler=wds.handlers.warn_and_continue, 
                             empty_check=False, nodesplitter=split_by_node)  # Explicitly set to None
                .with_epoch(1000)  # Ensure we don't run out of data
                .shuffle(1000)
                .decode("pil")
                .rename(image="jpg;jpeg;png")
                .map_dict(image=train_transform)
                .to_tuple("image")
                .batched(args.batch_size)
                # Ensure we get tensors, not lists
                .map(lambda batch_tuple: tuple(torch.stack(x) if isinstance(x, list) else x for x in batch_tuple))
            )
        else:
            # Single GPU setup
            dataset = (
                wds.WebDataset(args.dataset_path, handler=wds.handlers.warn_and_continue, empty_check=False)
                .with_epoch(1000)  # Ensure we don't run out of data
                .shuffle(1000)
                .decode("pil")
                .rename(image="jpg;jpeg;png")
                .map_dict(image=train_transform)
                .to_tuple("image")
                .batched(args.batch_size)
                # Ensure we get tensors, not lists
                .map(lambda batch_tuple: tuple(torch.stack(x) if isinstance(x, list) else x for x in batch_tuple))
            )
        
        train_loader = wds.WebLoader(
            dataset,
            batch_size=None,  # Batching is done by WebDataset
            shuffle=False,    # Shuffling is done by WebDataset
            num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
            pin_memory=True,
            worker_init_fn=None  # Disable WebDataset's default worker initialization
        )
        
        # Set epoch length - estimate based on number of shards
        if '{' in args.dataset_path:
            # Extract shard range from pattern like 'celeba_{00000..00040}.tar'
            import re
            pattern = r'\{(\d+)\.\.(\d+)\}'
            match = re.search(pattern, args.dataset_path)
            if match:
                start, end = match.groups()
                num_shards = int(end) - int(start) + 1
                # Use dataset_size from args if provided, otherwise use estimates
                if hasattr(args, 'dataset_size') and args.dataset_size is not None:
                    estimated_samples = args.dataset_size
                    logger.info(f"Using provided dataset size: {estimated_samples} samples")
                else:
                    # Rough estimate: 1000 samples per shard
                    estimated_samples = num_shards * 1000
                samples_per_gpu = estimated_samples // world_size
                # Set a reasonable length for the dataloader
                steps_per_epoch = samples_per_gpu // args.batch_size
                train_loader.length = steps_per_epoch
                logger.info(f"Estimated {estimated_samples} total samples, {samples_per_gpu} samples per GPU, {steps_per_epoch} steps per epoch")
            else:
                # Default length if pattern doesn't match
                train_loader.length = 1000 // args.batch_size
                logger.info(f"Could not determine dataset size, using default length of {train_loader.length} steps per epoch")
        else:
            # Single shard, use default length
            train_loader.length = 1000 // args.batch_size
            logger.info(f"Single shard dataset, using default length of {train_loader.length} steps per epoch")
    else:
        # Fallback to original ImagePaths implementation
        logger.info(f"Using ImagePaths with path: {args.dataset_path}")
        train_data = ImagePaths(args.dataset_path, size=args.image_size if hasattr(args, 'image_size') else 256)
        
        if is_distributed:
            train_sampler = DistributedSampler(
                train_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            train_loader = DataLoader(
                train_data, 
                batch_size=args.batch_size,
                shuffle=False,
                sampler=train_sampler,
                num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
                pin_memory=True,
                drop_last=True
            )
        else:
            train_loader = DataLoader(
                train_data, 
                batch_size=args.batch_size, 
                shuffle=True,
                num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
                pin_memory=True,
                drop_last=True
            )
    
    return train_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

# --------------------------------------------- #
#               Evaluation Utils
# --------------------------------------------- #

def calculate_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio between original and reconstructed images.
    
    Args:
        original: Original images tensor with values in range [-1, 1]
        reconstructed: Reconstructed images tensor with values in range [-1, 1]
        
    Returns:
        PSNR value as a scalar tensor
    """
    # Convert from [-1, 1] to [0, 1] range
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    # Calculate MSE
    mse = torch.mean((original - reconstructed) ** 2, dim=[1, 2, 3])  # Mean across C, H, W dimensions
    
    # Calculate PSNR
    psnr = 10 * torch.log10(1.0 / mse)  # max_pixel_value is 1.0 for normalized images
    
    return psnr


def load_validation_data(args, logger=None):
    """Load a small validation dataset for consistent evaluation.
    
    Args:
        args: Arguments containing dataset configuration
        logger: Optional logger for logging messages
        
    Returns:
        A DataLoader for the validation dataset
    """
    if logger is None:
        logger = logging.getLogger("vqgan_training")
    
    # Check if we should use WebDataset
    use_webdataset = args.dataset_path.endswith('.tar') or '{' in args.dataset_path
    
    # Get validation sample size from config
    val_samples = getattr(args, 'val_samples', 16)  # Default to 16 if not specified
    
    if use_webdataset and '{' in args.dataset_path:
        # Extract the base pattern from the dataset path
        base_pattern = args.dataset_path.split('{')[0]
        # Use the first shard for validation
        val_path = f"{base_pattern}00000.tar"
        
        # Check if distributed training is enabled
        is_distributed = dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0
        
        val_transform = create_webdataset_transform(size=args.image_size, is_train=False)
        
        # For multi-GPU training, only rank 0 needs validation data
        if is_distributed and rank != 0:
            logger.info(f"Rank {rank}: Skipping validation data loading (only needed on rank 0)")
            return None
            
        # Create validation dataset without nodesplitter for single-node
        val_dataset = (
            wds.WebDataset(val_path, handler=wds.handlers.warn_and_continue, 
                         empty_check=False, nodesplitter=split_by_node)  # No nodesplitter for single-node
            .with_epoch(1000)  # Ensure we don't run out of data
            .decode("pil")
            .rename(image="jpg;jpeg;png")
            .map_dict(image=val_transform)
            .to_tuple("image")
            .batched(val_samples)
            # Ensure we get tensors, not lists
            .map(lambda batch_tuple: tuple(torch.stack(x) if isinstance(x, list) else x for x in batch_tuple))
        )
        
        # Create validation loader
        val_loader = wds.WebLoader(
            val_dataset,
            batch_size=None,  # Batching is done by WebDataset
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Created validation dataset with {val_samples} samples from {val_path}")
        return val_loader
    elif not use_webdataset and os.path.isdir(args.dataset_path):
        # For ImagePaths dataset
        dataset = ImagePaths(args.dataset_path, size=args.image_size)
        val_subset_size = min(val_samples, len(dataset))
        
        # Create a small subset for validation
        indices = list(range(val_subset_size))
        val_dataset = torch.utils.data.Subset(dataset, indices)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_subset_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Created validation dataset with {val_subset_size} images from {args.dataset_path}")
        return val_loader
    else:
        logger.warning("Could not create validation dataset with the provided configuration")
        return None


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images: dict):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    new_sample = images["new_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(new_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()


def upload_ckpt(local_path, remote_path=None):
    """Upload checkpoint to B2 cloud storage (optional, requires b2 CLI)."""
    if remote_path is None:
        print("upload_ckpt: no remote_path specified, skipping")
        return
    import subprocess
    local_path = local_path.strip("/")
    remote_path = f'{remote_path}/{local_path.split("/")[-1]}'
    cmd = ["b2", "sync", local_path, remote_path]
    print(f"uploading ckpt: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"upload_ckpt failed (non-fatal): {e}")

"""
ImageNet NPZ dataset loader for flexible-length token training.
Loads pre-extracted latent codes from NPZ files with variable token budget support.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ImageNetNPZDataset(Dataset):
    """
    Dataset for loading pre-extracted ImageNet latent codes from NPZ files.
    Supports flexible token budgets for training.
    """

    def __init__(
        self,
        npz_path,
        vocab_size=65536,  # BSQ codebook size
        split='train',
        min_tokens=32,
        max_tokens=512,
        random_token_budget=True,
        fixed_token_budget=None,
    ):
        """
        Args:
            npz_path: Path to NPZ file or directory containing NPZ files
            vocab_size: Size of the vocabulary (codebook size)
            split: 'train' or 'val'
            min_tokens: Minimum number of tokens to use
            max_tokens: Maximum number of tokens to use (512 for full sequence)
            random_token_budget: If True, randomly sample token budget per item
            fixed_token_budget: If set, use this fixed number of tokens
        """
        # Find NPZ file
        if os.path.isdir(npz_path):
            npz_file = os.path.join(npz_path, f"{split}_latents.npz")
        else:
            npz_file = npz_path

        if not os.path.exists(npz_file):
            raise FileNotFoundError(f"NPZ file not found: {npz_file}")

        print(f"Loading {split} data from {npz_file}...")

        # Load NPZ file (memory-mapped for efficiency)
        data = np.load(npz_file, mmap_mode='r')

        # Extract data
        self.codes = data['indices']  # Shape: [N, 512] for full sequences
        self.names = data.get('names', None)
        self.num_samples = len(self.codes)

        # Token budget settings
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.random_token_budget = random_token_budget
        self.fixed_token_budget = fixed_token_budget

        # Special tokens - place them AFTER the vocabulary range
        # BSQ uses indices 0-65535, so special tokens start at 65536
        self.vocab_size = vocab_size
        self.pad_token_id = vocab_size + 2   # 65538 - Padding token
        self.bos_token_id = vocab_size       # 65536 - Beginning of sequence
        self.eos_token_id = vocab_size + 1   # 65537 - End of sequence

        print(f"Loaded {self.num_samples} samples")
        print(f"Full sequence length: {self.codes.shape[1]} tokens")
        print(f"Token budget range: [{self.min_tokens}, {self.max_tokens}]")
        print(f"Random budget: {self.random_token_budget}")
        print(f"Special tokens - PAD: {self.pad_token_id}, BOS: {self.bos_token_id}, EOS: {self.eos_token_id}")

        # Validate codes are within vocabulary (0 to vocab_size-1)
        max_code = np.max(self.codes)
        if max_code >= self.vocab_size:
            raise ValueError(f"Code {max_code} exceeds vocabulary size {self.vocab_size}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns a sample with flexible token budget.
        """
        # Get full sequence of codes for this image
        full_codes = self.codes[idx]  # Shape: [512]

        # Determine token budget for this sample
        if self.fixed_token_budget is not None:
            token_budget = self.fixed_token_budget
        elif self.random_token_budget:
            # Random sampling during training
            token_budget = np.random.randint(self.min_tokens, self.max_tokens + 1)
        else:
            # Use max tokens
            token_budget = self.max_tokens

        # Truncate to token budget
        codes = full_codes[:token_budget]

        # Convert to tensor
        sequence = torch.from_numpy(codes.astype(np.int64))

        # Add BOS and EOS tokens for autoregressive training
        bos_token = torch.tensor([self.bos_token_id], dtype=torch.long)
        eos_token = torch.tensor([self.eos_token_id], dtype=torch.long)

        # Create full sequence: [BOS, seq[0], seq[1], ..., seq[n], EOS]
        full_sequence = torch.cat([bos_token, sequence, eos_token])

        # For autoregressive training:
        # Input:  [BOS, seq[0], seq[1], ..., seq[n]]
        # Target: [seq[0], seq[1], ..., seq[n], EOS]
        input_ids = full_sequence[:-1]
        target_ids = full_sequence[1:]

        # Store actual length (before padding in collate)
        seq_length = len(input_ids)

        result = {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'seq_length': seq_length,
            'token_budget': token_budget,
        }

        # Add image name if available
        if self.names is not None:
            result['image_name'] = self.names[idx]

        return result


def collate_fn_imagenet(batch, pad_token_id=65538):
    """
    Custom collate function for variable length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Extract components
    input_ids_list = [item['input_ids'] for item in batch]
    target_ids_list = [item['target_ids'] for item in batch]
    seq_lengths = torch.tensor([item['seq_length'] for item in batch])
    token_budgets = torch.tensor([item['token_budget'] for item in batch])

    # Pad sequences to max length in this batch
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_id)
    target_ids = pad_sequence(target_ids_list, batch_first=True, padding_value=pad_token_id)

    # Create attention mask (1 for real tokens, 0 for padding)
    batch_size, max_len = input_ids.shape
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        attention_mask[i, :length] = 1

    result = {
        'input_ids': input_ids,        # [batch_size, max_seq_len_in_batch]
        'target_ids': target_ids,      # [batch_size, max_seq_len_in_batch]
        'attention_mask': attention_mask,  # [batch_size, max_seq_len_in_batch]
        'seq_lengths': seq_lengths,    # [batch_size]
        'token_budgets': token_budgets,  # [batch_size] - for logging
    }

    # Add image names if available
    if 'image_name' in batch[0]:
        result['image_names'] = [item['image_name'] for item in batch]

    return result


def build_imagenet_npz(args, **kwargs):
    """
    Build ImageNet NPZ dataset from arguments.
    """
    # Get paths and settings from args - check kwargs first (from config), then args
    npz_path = kwargs.get('npz_path', getattr(args, 'npz_path', './imagenet_latents'))
    split = kwargs.get('split', getattr(args, 'split', 'train'))

    # Token budget settings - check kwargs first (from config), then args
    min_tokens = kwargs.get('min_tokens', getattr(args, 'min_tokens', 32))
    max_tokens = kwargs.get('max_tokens', getattr(args, 'max_tokens', 512))
    random_token_budget = kwargs.get('random_token_budget', getattr(args, 'random_token_budget', True))
    fixed_token_budget = kwargs.get('fixed_token_budget', getattr(args, 'fixed_token_budget', None))

    # Vocab settings
    vocab_size = kwargs.get('vocab_size', getattr(args, 'vocab_size', 65536))

    # Override for validation - always use full tokens
    if split == 'val':
        random_token_budget = False
        fixed_token_budget = max_tokens

    return ImageNetNPZDataset(
        npz_path=npz_path,
        vocab_size=vocab_size,
        split=split,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        random_token_budget=random_token_budget,
        fixed_token_budget=fixed_token_budget,
    )


# For testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test ImageNet NPZ dataset loader')
    parser.add_argument('--npz_path', type=str,
                        default=os.environ.get('IMAGENET_NPZ_PATH', './imagenet_latents'),
                        help='Path to NPZ file or directory')
    parser.add_argument('--vocab_size', type=int, default=65536)
    parser.add_argument('--min_tokens', type=int, default=32)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--random_token_budget', action='store_true', default=True)
    parser.add_argument('--fixed_token_budget', type=int, default=None)
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])

    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"Warning: NPZ path {args.npz_path} does not exist")
        print("Set --npz_path or IMAGENET_NPZ_PATH environment variable")
        exit(1)

    dataset = build_imagenet_npz(args)

    print(f"\nDataset size: {len(dataset)}")

    # Test a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  Target shape: {sample['target_ids'].shape}")
        print(f"  Token budget: {sample['token_budget']}")
        print(f"  First 10 input tokens: {sample['input_ids'][:10]}")

    # Test collate function if we have enough samples
    if len(dataset) >= 4:
        batch = [dataset[i] for i in range(4)]
        collated = collate_fn_imagenet(batch)
        print(f"\nCollated batch:")
        print(f"  Input shape: {collated['input_ids'].shape}")
        print(f"  Attention mask shape: {collated['attention_mask'].shape}")
        print(f"  Token budgets: {collated['token_budgets']}")
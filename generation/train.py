# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import yaml
import wandb
import numpy as np
from PIL import Image

from generation.utils.logger import create_logger
from generation.utils.distributed import init_distributed_mode
from generation.utils.ema import update_ema, requires_grad
from generation.dataset.build import build_dataset
from generation.dataset.imagenet_npz import collate_fn_imagenet
from generation.models.gpt import GPT_models
from generation.decode_utils import reconstruct_from_channelwise_indices, denorm_to_uint8

from models import get_model_class
from utils.checkpoint import load_config as load_decoder_config, load_checkpoint, clean_state_dict
from omegaconf import OmegaConf

#################################################################################
#                             Functions for decoder model                        #
#################################################################################


def load_decoder_model(ckpt_path, cfg, device):
    """Load a decoder model using the consolidated checkpoint utility."""
    return load_checkpoint(ckpt_path, cfg, device=device)


#################################################################################
#                         Position-Weighted Loss for Channelwise AR             #
#################################################################################

def compute_position_weights(seq_len, weight_type="none", alpha=1.0, device="cuda"):
    """
    Compute per-position loss weights for channelwise autoregressive training.

    In channelwise AR, early tokens encode coarse global structure and errors
    cascade to all later tokens. This weighting forces the model to prioritize
    getting early (structural) channels right.

    Args:
        seq_len: length of the target sequence (e.g. 257 = 256 channels + EOS)
        weight_type: "none" | "linear" | "exponential"
        alpha: strength of the weighting (0 = uniform, higher = more front-loaded)
        device: torch device

    Returns:
        weights: tensor of shape [seq_len], all 1.0 if weight_type=="none"
    """
    if weight_type == "none":
        return torch.ones(seq_len, device=device)

    # Position indices normalized to [0, 1]
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    t = pos / max(seq_len - 1, 1)  # 0.0 at first position, 1.0 at last

    if weight_type == "linear":
        # Linear decay: position 0 gets (1 + alpha), last position gets 1.0
        weights = 1.0 + alpha * (1.0 - t)
    elif weight_type == "exponential":
        # Exponential decay: position 0 gets exp(alpha), last position gets 1.0
        weights = torch.exp(alpha * (1.0 - t))
    else:
        raise ValueError(f"Unknown position_weight_type: {weight_type}. Use 'none', 'linear', or 'exponential'.")

    return weights


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def visualize_flexible_generation(model, decoder_model, dataset, device, args, rank=0, wandb_enabled=False, step=0):
    """
    Generate and visualize images at different token budgets.
    Shows the quality progression from low to high token counts.
    """
    try:
        was_training = model.training
        model.eval()

        # Calculate latent size from config
        latent_size = args.image_size // args.downsample_size

        # Get visualization settings from args
        num_samples = getattr(args, 'num_viz_samples', 4)
        token_budgets = getattr(args, 'viz_token_budgets', [64, 128, 256, 512])

        # Special tokens
        bos_token_id = dataset.bos_token_id
        eos_token_id = dataset.eos_token_id
        vocab_size = dataset.vocab_size

        all_images = {}  # Store images for each budget

        for budget in token_budgets:
            budget_images = []

            for i in range(min(num_samples, len(dataset))):
                # Generate with fixed length = budget
                generated = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

                # Generate exactly 'budget' tokens
                for j in range(budget):
                    dummy_cond = torch.zeros(1, dtype=torch.long, device=device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits, _ = model(idx=generated, cond_idx=dummy_cond)

                    # Sample next token
                    temperature = 1.0
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Stop if EOS generated
                    if next_token.item() == eos_token_id:
                        break

                    generated = torch.cat([generated, next_token], dim=1)

                # Process generated tokens
                gen_indices = generated[0].detach().cpu()
                if gen_indices[0] == bos_token_id:
                    gen_indices = gen_indices[1:]
                if len(gen_indices) > 0 and gen_indices[-1] == eos_token_id:
                    gen_indices = gen_indices[:-1]

                # Filter special tokens
                gen_indices = gen_indices[gen_indices < vocab_size - 3]

                # Reconstruct image
                if len(gen_indices) > 0:
                    gen_indices = gen_indices[:budget]  # Ensure we use exactly 'budget' tokens
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        gen_image, _ = reconstruct_from_channelwise_indices(
                            decoder_model,
                            gen_indices,
                            latent_hw=(latent_size, latent_size),
                            total_channels=None
                        )
                    gen_image = denorm_to_uint8(gen_image)
                    budget_images.append(gen_image)

            all_images[budget] = budget_images

        # Create visualization grid: rows = samples, columns = token budgets
        if all_images and all(len(imgs) > 0 for imgs in all_images.values()):
            img_h, img_w = next(iter(all_images.values()))[0].shape[:2]
            grid_h = num_samples * img_h
            grid_w = len(token_budgets) * img_w
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

            for col_idx, budget in enumerate(token_budgets):
                for row_idx in range(min(num_samples, len(all_images[budget]))):
                    grid[row_idx*img_h:(row_idx+1)*img_h,
                         col_idx*img_w:(col_idx+1)*img_w] = all_images[budget][row_idx]

            # Log to wandb
            if rank == 0 and wandb_enabled:
                wandb_dict = {
                    "flexible_generation": wandb.Image(
                        grid,
                        caption=f"Token budgets: {token_budgets} (step {step})"
                    ),
                    "train_step": step
                }

                # Also log individual images for each budget
                for budget in token_budgets:
                    budget_grid = np.concatenate(all_images[budget], axis=0)
                    wandb_dict[f"generation_{budget}_tokens"] = wandb.Image(
                        budget_grid,
                        caption=f"{budget} tokens (step {step})"
                    )

                wandb.log(wandb_dict)

        if was_training:
            model.train()

    except Exception as e:
        print(f"Error in visualization: {e}")
        if was_training:
            model.train()


@torch.no_grad()
def log_reconstructions_to_wandb(model, decoder_model, dataset, device, args, rank=0, wandb_enabled=False, num_samples=4, step=0):
    """Generate images using the model and log reconstructions to wandb."""
    try:
        was_training = model.training
        model.eval()

        # Calculate latent size from config
        latent_size = args.image_size // args.downsample_size

        # Special tokens from dataset
        bos_token_id = dataset.bos_token_id  # 63999
        eos_token_id = dataset.eos_token_id  # 63998
        pad_token_id = dataset.pad_token_id  # 64000
        vocab_size = dataset.vocab_size      # 64000
        
        # Create lists for ground truth and generated images (on CPU to save GPU memory)
        gt_images = []
        gen_images = []
        
        for i in range(min(num_samples, len(dataset))):
            # Get a sample from the dataset for ground truth
            sample = dataset[i]
            
            # === Ground Truth Reconstruction ===
            # Get the actual FSQ indices (without BOS/EOS tokens)
            gt_indices = sample['input_ids'][1:sample['seq_length']-1]  # Remove BOS, stop before last token
            
            # If the sequence has EOS, remove it
            if len(gt_indices) > 0 and gt_indices[-1] == eos_token_id:
                gt_indices = gt_indices[:-1]
            
            # Reconstruct ground truth image
            if len(gt_indices) > 0:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    gt_image, _ = reconstruct_from_channelwise_indices(
                        decoder_model,
                        gt_indices,
                        latent_hw=(latent_size, latent_size),
                        total_channels=None
                    )
                gt_image = denorm_to_uint8(gt_image)  # This already moves to CPU
                gt_images.append(gt_image)
                del gt_image  # Explicit cleanup
            
            # === Model-Generated Reconstruction ===
            # Start with BOS token
            generated = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
            max_length = min(256, dataset.max_seq_length)  # Reduced max length to save memory
            
            # Generate autoregressively
            for j in range(max_length - 1):
                # Create dummy conditioning (since cls_token_num=0)
                dummy_cond = torch.zeros(1, dtype=torch.long, device=device)
                
                # Get model predictions
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = model(idx=generated, cond_idx=dummy_cond)
                
                # Sample from the last position - immediately detach and work on CPU
                next_token_logits = logits[:, -1, :].detach()
                
                # Apply temperature sampling
                temperature = 1.0
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Cleanup intermediate tensors
                del logits, next_token_logits, probs
                
                # Stop if we generate EOS token
                if next_token.item() == eos_token_id:
                    del next_token
                    break
                del next_token
            
            # Move generated to CPU immediately to free GPU memory
            gen_indices = generated[0].detach().cpu()
            del generated  # Free GPU memory
            
            # Remove BOS token at start
            if gen_indices[0] == bos_token_id:
                gen_indices = gen_indices[1:]
            # Remove EOS token at end if present
            if len(gen_indices) > 0 and gen_indices[-1] == eos_token_id:
                gen_indices = gen_indices[:-1]
            
            # Filter out any special tokens
            gen_indices = gen_indices[(gen_indices < vocab_size - 2)]
            
            # Reconstruct generated image
            if len(gen_indices) > 0:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    gen_image, _ = reconstruct_from_channelwise_indices(
                        decoder_model, 
                        gen_indices,
                        latent_hw=(latent_size, latent_size),
                        total_channels=None
                    )
                gen_image = denorm_to_uint8(gen_image)  # Already moves to CPU
                gen_images.append(gen_image)
                del gen_image  # Explicit cleanup
            
            del gen_indices  # Cleanup
        
        # Create comparison grid on CPU
        if len(gt_images) > 0 and len(gen_images) > 0:
            img_h, img_w = gt_images[0].shape[:2]
            grid_h = min(num_samples, len(gt_images)) * img_h
            grid_w = 2 * img_w
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            
            for idx in range(min(len(gt_images), len(gen_images))):
                # Ground truth on the left
                grid[idx*img_h:(idx+1)*img_h, 0:img_w] = gt_images[idx]
                # Generated on the right
                grid[idx*img_h:(idx+1)*img_h, img_w:2*img_w] = gen_images[idx]
            
            # Convert grid to PIL Image
            pil_grid = Image.fromarray(grid)
            
            # Log comparison grid to wandb
            if rank == 0 and wandb_enabled:
                wandb_image = wandb.Image(pil_grid, caption=f"Step {step}: GT (left) vs Generated (right)")

                # Log with commit=False to batch with other metrics
                wandb.log({
                    "reconstructions": wandb_image,
                    "reconstruction_step": step
                }, commit=False)

                # Cleanup wandb image
                del wandb_image

            # Cleanup
            del grid, pil_grid
        
        # Clear lists
        del gt_images, gen_images
        
        # Clear GPU cache after reconstruction
        torch.cuda.empty_cache()
        
    finally:
        # Always restore model state
        if was_training:
            model.train()


def load_config_from_yaml(yaml_path, args):
    """Load configuration from YAML file and merge with argparse args."""
    if yaml_path is None:
        return args, {}

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle defaults inheritance (simple implementation)
    final_config = {}
    if 'defaults' in config:
        defaults_list = config.get('defaults', [])
        if isinstance(defaults_list, str):
            defaults_list = [defaults_list]

        # Load base configs first
        config_dir = os.path.dirname(yaml_path)
        for default in defaults_list:
            if default.startswith('-'):
                default = default[1:].strip()
            base_path = os.path.join(config_dir, f"{default}.yaml")
            if os.path.exists(base_path):
                with open(base_path, 'r') as f:
                    base_config = yaml.safe_load(f)
                    final_config.update(base_config)
                print(f"Loaded base config from: {base_path}")

    # Override with current config (excluding defaults key)
    for key, value in config.items():
        if key != 'defaults':
            final_config[key] = value

    # Convert YAML keys to match argparse attribute names (replace - with _)
    for key, value in final_config.items():
        if value is not None:  # Only override if value is not None in YAML
            attr_name = key.replace('-', '_')
            if hasattr(args, attr_name):
                setattr(args, attr_name, value)
            else:
                # Create the attribute if it doesn't exist
                setattr(args, attr_name, value)

    return args, final_config
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args, yaml_config={}):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    
    # Handle both distributed and single GPU modes
    if args.distributed:
        assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        world_size = dist.get_world_size()
    else:
        # Single GPU mode
        rank = 0
        device = args.gpu
        seed = args.global_seed
        world_size = 1
    
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
        # Check if we're resuming in the same folder
        if args.gpt_ckpt and args.resume_same_folder:
            # Extract experiment directory from checkpoint path
            # Expected format: results/026-GPT-Nano/checkpoints/latest.pt
            checkpoint_parent = os.path.dirname(os.path.dirname(args.gpt_ckpt))
            if os.path.exists(checkpoint_parent) and 'results' in checkpoint_parent:
                experiment_dir = checkpoint_parent
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
                model_string_name = os.path.basename(experiment_dir).split('-', 1)[-1]
                experiment_index = int(os.path.basename(experiment_dir).split('-')[0])  # Convert to int
                
                # Extract WandB run ID if it exists
                wandb_dir = os.path.join(experiment_dir, 'wandb')
                if os.path.exists(wandb_dir):
                    run_folders = glob(os.path.join(wandb_dir, 'run-*'))
                    if run_folders:
                        latest_run = max(run_folders, key=os.path.getmtime)
                        run_name = os.path.basename(latest_run)
                        if '-' in run_name:
                            wandb_run_id = run_name.split('-')[-1]
                            os.environ['WANDB_RUN_ID'] = wandb_run_id
                            os.environ['WANDB_RESUME'] = 'must'
                            print(f"Resuming WandB run: {wandb_run_id}")
                
                logger = create_logger(experiment_dir)
                logger.info(f"Resuming in existing directory: {experiment_dir}")
            else:
                # Fallback to creating new folder if path doesn't match expected format
                os.makedirs(args.results_dir, exist_ok=True)
                experiment_index = len(glob(f"{args.results_dir}/*"))
                model_string_name = args.gpt_model.replace("/", "-")
                experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
                checkpoint_dir = f"{experiment_dir}/checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger = create_logger(experiment_dir)
                logger.info(f"Created new experiment directory: {experiment_dir}")
        else:
            # Normal behavior - create new folder
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.gpt_model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    
    # Initialize wandb on rank 0 only
    wandb_enabled = False
    if rank == 0:
        # Combine args and yaml config for wandb
        wandb_config = vars(args).copy()
        wandb_config['yaml_config'] = yaml_config  # Add original yaml config
        wandb_config['config_file'] = args.config if hasattr(args, 'config') else None
        
        wandb_entity = os.environ.get('WANDB_ENTITY', getattr(args, 'wandb_entity', None))
        wandb_project = getattr(args, 'wandb_project', 'channeltok')

        wandb_run_name = getattr(args, 'wandb_name', None)
        if not wandb_run_name:
            wandb_run_name = f"{model_string_name}-{time.strftime('%Y%m%d-%H%M%S')}"

        if getattr(args, 'wandb_enabled', True):
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config,
                dir=experiment_dir,
                entity=wandb_entity,
            )
            wandb_enabled = True
            logger.info(f"Wandb initialized: project='{wandb_project}', name='{wandb_run_name}'")
        else:
            wandb_enabled = False
            logger.info("Wandb disabled via config")  


    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size

    print("using GPT model: ",args.gpt_model)
    
    # For variable length training, we need extra tokens:
    # Original vocab: 0 to 63997 (FSQ codes)
    # Special tokens:
    # - 63998 = EOS token
    # - 63999 = BOS token  
    # - 64000 = PAD token
    # So model needs vocab_size + 1 embeddings (0 to 64000 inclusive = 64001 total)
    if hasattr(args, 'pad_token_id'):
        model_vocab_size = args.pad_token_id + 1  # 64000 + 1 = 64001
        print(f"Using model_vocab_size={model_vocab_size} (pad_token_id={args.pad_token_id})")
    else:
        model_vocab_size = args.vocab_size
        print(f"Using default model_vocab_size={model_vocab_size}")
    
    # For variable-length sequences, use max_sequence_length for positional embeddings
    # For fixed 2D grids, use target_seq_length if it's a perfect square
    target_seq_length = getattr(args, 'target_seq_length', None)
    max_seq_length = getattr(args, 'max_sequence_length', 512)
    
    if target_seq_length and int(target_seq_length ** 0.5) ** 2 == target_seq_length:
        # Use 2D positional encoding for perfect square grids
        block_size = target_seq_length  # Use target length (e.g., 256 for 16x16)
        print(f"Using 2D positional encoding with block_size={block_size} (grid: {int(target_seq_length ** 0.5)}x{int(target_seq_length ** 0.5)})")
    else:
        # Use 1D positional encoding for variable-length sequences
        # Use max_sequence_length to ensure we can handle any sequence up to that length
        block_size = max_seq_length
        print(f"Using 1D positional encoding with block_size={block_size} for variable-length sequences")
    
    # Get use_learned_pos_emb from args (defaults to False if not set)
    use_learned_pos_emb = getattr(args, 'use_learned_pos_emb', False)
    
    model = GPT_models[args.gpt_model](
        vocab_size=model_vocab_size,
        block_size=block_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        use_learned_pos_emb=use_learned_pos_emb,  # Pass learned pos emb flag
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data - pass yaml_config as kwargs to get config values
    dataset = build_dataset(args, **yaml_config)

    # Setup LR scheduler (if enabled)
    scheduler = None
    use_lr_schedule = getattr(args, 'use_lr_schedule', False)
    if use_lr_schedule:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        # Calculate total steps
        steps_per_epoch = len(dataset) // args.global_batch_size
        total_steps = args.epochs * steps_per_epoch
        warmup_fraction = getattr(args, 'warmup_fraction', 0.1)
        warmup_steps = int(total_steps * warmup_fraction)
        min_lr_factor = getattr(args, 'min_lr_factor', 0.01)
        min_lr = args.lr * min_lr_factor

        logger.info(f"Setting up LR scheduler:")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps} ({warmup_fraction*100:.0f}%)")
        logger.info(f"  Peak LR: {args.lr}")
        logger.info(f"  Min LR: {min_lr}")

        # Warmup scheduler: 0 → peak_lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-8 / args.lr,  # Start from very small LR
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Cosine annealing: peak_lr → min_lr
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr
        )

        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        logger.info("LR scheduler initialized successfully")

    # Get pad_token_id from args or dataset
    pad_token_id = getattr(args, 'pad_token_id', 64000)

    # Create custom collate function
    collate_fn = lambda batch: collate_fn_imagenet(batch, pad_token_id=pad_token_id)
    
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        batch_size = int(args.global_batch_size // world_size)
        shuffle = False
    else:
        # Single GPU mode - no distributed sampler
        sampler = None
        batch_size = args.global_batch_size
        shuffle = True
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn  # Use custom collate function
    )
    logger.info(f"Dataset contains {len(dataset):,} FSQ samples")

    # Prepare models for training:
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        #if learning rate is different in the config, override the loaded learning rate
        lr_overridden = args.lr != checkpoint["args"].lr
        if lr_overridden:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            logger.info(f"Overrode optimizer LR from {checkpoint['args'].lr} to {args.lr}")

        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))

        # Handle scheduler: if LR was overridden, create fresh schedule (warm restart)
        # Otherwise restore from checkpoint
        if lr_overridden and scheduler is not None:
            logger.info(f"LR overridden -> creating fresh scheduler (warm restart to {args.lr})")
            # Scheduler was already created fresh above with new LR, just use it as-is
        elif scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("Loaded scheduler state from checkpoint")
        elif scheduler is not None:
            logger.info("No scheduler state in checkpoint, will resume from current step")
            for _ in range(train_steps):
                scheduler.step()

        del checkpoint

        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        


    decoder_config = load_decoder_config(args.decoder_config)
    decoder_model = load_decoder_model(args.decoder_ckpt, decoder_config, 'cuda')
    decoder_total_channels = decoder_config.model.latent_dim  # 256 for 16K, 512 for 65K
    logger.info(f"Decoder latent_dim (total_channels): {decoder_total_channels}")

    # Wrap model in DDP only for distributed training
    if args.distributed:
        model = DDP(model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = model.to(device)
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    # Setup position-weighted loss (for channelwise AR)
    # Weights are computed lazily on first batch to derive length from actual
    # target_ids.shape[1], avoiding off-by-one from config's target_seq_length.
    pw_type = getattr(args, 'position_weight_type', 'none')
    pw_alpha = getattr(args, 'position_weight_alpha', 1.0)
    position_weights = None  # computed on first batch if pw_type != 'none'
    if pw_type != 'none':
        logger.info(f"Position-weighted loss: type={pw_type}, alpha={pw_alpha} (weights computed on first batch)")
    else:
        logger.info("Position-weighted loss: disabled (uniform weights)")

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_perplexity = 0  # Track perplexity for generation quality
    latest_unweighted_perplexity = None  # Boundary-only metric when position weighting is enabled
    running_accuracy = 0  # Also track token-level accuracy for debugging
    start_time = time.time()

    grad_accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
    logger.info(f"Training for {args.epochs} epochs (gradient_accumulation_steps={grad_accum_steps})...")
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
            # Ensure all ranks start the epoch together
            dist.barrier()
            if rank == 0:
                logger.info(f"All ranks synchronized for epoch {epoch}")
        
        logger.info(f"Beginning epoch {epoch}...")
        
        # Add batch counter for debugging
        batch_count = 0
        
        for batch in loader:
            batch_count += 1
            # Your dataset returns dict with input_ids, target_ids, attention_mask, etc.
            input_ids = batch['input_ids'].to(device, non_blocking=True)  # [batch_size, seq_len]
            target_ids = batch['target_ids'].to(device, non_blocking=True)  # [batch_size, seq_len]
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)  # [batch_size, seq_len]
            seq_lengths = batch['seq_lengths'].to(device, non_blocking=True)  # [batch_size]
            
            with torch.amp.autocast(dtype=ptdtype):
                # Unconditional training: provide dummy conditioning tensor
                # Since cls_token_num=0, this will be sliced to empty anyway
                batch_size = input_ids.shape[0]
                dummy_cond = torch.zeros(batch_size, dtype=torch.long, device=device)

                # Lazily compute position weights on first batch (correct length)
                if position_weights is None and pw_type != 'none':
                    actual_seq_len = target_ids.shape[1]
                    position_weights = compute_position_weights(actual_seq_len, pw_type, pw_alpha, device=device)
                    if rank == 0:
                        print(f"Position weights computed: len={actual_seq_len}, "
                              f"range=[{position_weights.min():.3f}, {position_weights.max():.3f}]")
                        print(f"  First 5: {position_weights[:5].tolist()}")
                        print(f"  Last 5:  {position_weights[-5:].tolist()}")

                # Build the valid mask: attention_mask * position_weights
                # Position weights emphasize early channels for channelwise AR
                valid_mask = attention_mask.float()
                if position_weights is not None:
                    valid_mask = valid_mask * position_weights.unsqueeze(0)

                logits, loss = model(idx=input_ids, cond_idx=dummy_cond, targets=target_ids, valid=valid_mask)

            # backward pass, with gradient scaling if training in fp16
            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            # Only step optimizer on accumulation boundaries
            if batch_count % grad_accum_steps == 0:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)

                # Step LR scheduler if enabled
                if scheduler is not None:
                    scheduler.step()

                if args.ema:
                    if args.distributed:
                        update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)
                    else:
                        update_ema(ema, model._orig_mod if not args.no_compile else model)

                # Log loss values:
                running_loss += loss.item()

                # Log reconstructions periodically
                if args.log_reconstructions and train_steps % args.reconstruction_freq == 0 and rank == 0:
                    logger.info(f"Logging {args.num_reconstruction_samples} reconstructed images to wandb...")
                    # Use the actual model, unwrapping DDP if necessary
                    actual_model = model.module if args.distributed else model
                    if not args.no_compile:
                        actual_model = actual_model._orig_mod

                    log_reconstructions_to_wandb(
                        actual_model,
                        decoder_model,
                        dataset,
                        device,
                        args,
                        rank=rank,
                        wandb_enabled=wandb_enabled,
                        num_samples=args.num_reconstruction_samples,
                        step=train_steps
                    )

                # Calculate metrics: accuracy every step, perplexity at log boundaries
                with torch.no_grad():
                    # Token-level accuracy (cheap — just argmax compare)
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == target_ids) & (attention_mask > 0)
                    accuracy = correct.float().sum() / attention_mask.sum()
                    running_accuracy += accuracy.item()

                    # Perplexity: with position weighting, compute true unweighted CE
                    # only on log boundaries to avoid an extra CE pass every step.
                    if position_weights is not None:
                        if (train_steps + 1) % args.log_every == 0:
                            uw_mask_flat = attention_mask.float().view(-1)
                            loss_all_uw = F.cross_entropy(
                                logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='none'
                            )
                            unweighted_loss = (loss_all_uw * uw_mask_flat).sum() / max(uw_mask_flat.sum(), 1)
                            latest_unweighted_perplexity = torch.exp(unweighted_loss).item()
                    else:
                        perplexity = torch.exp(loss).item()
                        running_perplexity += perplexity

                log_steps += 1
                train_steps += 1
            
            # Periodic GPU cache clearing to prevent memory fragmentation
            if train_steps % 5000 == 0:
                torch.cuda.empty_cache()
                if rank == 0:
                    # Log memory usage
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    logger.info(f"Step {train_steps}: GPU {device} Memory - {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                
                # Add distributed barrier to ensure all ranks reach this point together
                if args.distributed:
                    if rank == 0:
                        #logger.info(f"Rank {rank}: Waiting for all ranks at logging step {train_steps}")
                        pass
                    dist.barrier()
                    if rank == 0:
                        #logger.info(f"Rank {rank}: All ranks synchronized for logging")
                        pass
                
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if args.distributed:
                    try:
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                        avg_loss = avg_loss.item() / world_size
                    except Exception as e:
                        logger.error(f"Rank {rank}: all_reduce failed: {e}")
                        avg_loss = avg_loss.item()  # Fallback to local loss
                else:
                    avg_loss = avg_loss.item()
                
                # Reduce perplexity over all processes
                if position_weights is not None:
                    if latest_unweighted_perplexity is None:
                        avg_perplexity = float("nan")
                    else:
                        avg_perplexity_t = torch.tensor(latest_unweighted_perplexity, device=device)
                        if args.distributed:
                            dist.all_reduce(avg_perplexity_t, op=dist.ReduceOp.SUM)
                            avg_perplexity = avg_perplexity_t.item() / world_size
                        else:
                            avg_perplexity = avg_perplexity_t.item()
                else:
                    avg_perplexity_t = torch.tensor(running_perplexity / log_steps, device=device)
                    if args.distributed:
                        dist.all_reduce(avg_perplexity_t, op=dist.ReduceOp.SUM)
                        avg_perplexity = avg_perplexity_t.item() / world_size
                    else:
                        avg_perplexity = avg_perplexity_t.item()

                # Reduce accuracy over all processes
                avg_accuracy = torch.tensor(running_accuracy / log_steps, device=device)
                if args.distributed:
                    dist.all_reduce(avg_accuracy, op=dist.ReduceOp.SUM)
                    avg_accuracy = avg_accuracy.item() / world_size
                else:
                    avg_accuracy = avg_accuracy.item()

                logger.info(f"(step={train_steps:07d}) Loss: {avg_loss:.4f}, Perplexity: {avg_perplexity:.1f}, Acc: {avg_accuracy:.3f}, Steps/Sec: {steps_per_sec:.2f}")

                # Log to wandb
                if rank == 0 and wandb_enabled:
                    wandb_dict = {
                        "train/loss": avg_loss,
                        "train/perplexity": avg_perplexity,  # always unweighted CE perplexity
                        "train/accuracy": avg_accuracy,
                        "train/steps_per_sec": steps_per_sec,
                        "train/epoch": epoch,
                        "train/step": train_steps,
                        "train/learning_rate": optimizer.param_groups[0]['lr']
                    }
                    # When position weighting is active, avg_loss is the weighted loss
                    # (used for backprop) and perplexity is from unweighted CE (comparable)
                    if position_weights is not None:
                        wandb_dict["train/weighted_loss"] = avg_loss
                    wandb.log(wandb_dict)

                    # Visualize generation at different token budgets
                    visualize_every = getattr(args, 'visualize_every', 1000)
                    if visualize_every > 0 and train_steps % visualize_every == 0:
                        logger.info(f"Generating visualization at step {train_steps}...")
                        visualize_flexible_generation(
                            model.module if args.distributed else model,
                            decoder_model,
                            dataset,
                            device,
                            args,
                            rank=rank,
                            wandb_enabled=wandb_enabled,
                            step=train_steps
                        )
                
                # Reset monitoring variables:
                running_loss = 0
                running_perplexity = 0
                latest_unweighted_perplexity = None
                running_accuracy = 0
                log_steps = 0
                start_time = time.time()

        # Save checkpoint at end of epoch:
        # Use ckpt_every to compute epoch interval (save every N epochs, minimum 1)
        ckpt_every = getattr(args, 'ckpt_every', 5000)
        steps_per_epoch = max(len(loader), 1)
        save_every_n_epochs = max(ckpt_every // steps_per_epoch, 1)
        if rank == 0 and (epoch % save_every_n_epochs == 0 or epoch == args.epochs - 1):
            if args.distributed:
                if not args.no_compile:
                    model_weight = model.module._orig_mod.state_dict()
                else:
                    model_weight = model.module.state_dict()
            else:
                if not args.no_compile:
                    model_weight = model._orig_mod.state_dict()
                else:
                    model_weight = model.state_dict()  
            checkpoint = {
                "model": model_weight,
                "optimizer": optimizer.state_dict(),
                "steps": train_steps,
                "epoch": epoch,
                "args": args
            }
            if args.ema:
                checkpoint["ema"] = ema.state_dict()
            if scheduler is not None:
                checkpoint["scheduler"] = scheduler.state_dict()
            if not args.no_local_save:
                checkpoint_path = f"{checkpoint_dir}/latest.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path} (epoch {epoch}, step {train_steps})")
            
            cloud_checkpoint_path = f"{cloud_checkpoint_dir}/latest.pt"
            torch.save(checkpoint, cloud_checkpoint_path)
            logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path} (epoch {epoch}, step {train_steps})")

        if args.distributed:
            dist.barrier()
        logger.info(f"Completed epoch {epoch}")

    model.eval()  # important! This disables randomized embedding dropout

    # Clean up wandb
    if rank == 0 and wandb_enabled:
        wandb.finish()

    logger.info("Done!")
    if args.distributed:
        dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--code-path", type=str, required=False, help="Path to training data")
    parser.add_argument("--cloud-save-path", type=str, required=False, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_npz')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    # Add missing arguments for variable length sequences
    parser.add_argument("--max-sequence-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--target-seq-length", type=int, default=512, help="Target sequence length for padding")
    parser.add_argument("--pad-token-id", type=int, default=64000, help="Padding token ID")
    parser.add_argument("--bos-token-id", type=int, default=63999, help="Beginning of sequence token ID")
    parser.add_argument("--eos-token-id", type=int, default=63998, help="End of sequence token ID")
    parser.add_argument("--resume-same-folder", action='store_true', help="Resume training in the same folder (preserves WandB run)")
    
    #add arguments for decoder model.  
    parser.add_argument("--decoder-ckpt", type=str, default='../ckpts/vqganae_fsq_channel_512/checkpoints/best.ckpt')
    parser.add_argument("--decoder-config", type=str, default='../configs/exp/vqgan_ae_fsq_ch_512_runpod.yaml')
    parser.add_argument("--log-reconstructions", action='store_true', help="Log image reconstructions to wandb")
    parser.add_argument("--reconstruction-freq", type=int, default=1000, help="How often to log reconstructions (in steps)")
    parser.add_argument("--num-reconstruction-samples", type=int, default=4, help="Number of images to reconstruct and log")
    
    
    args = parser.parse_args()
    
 
    args, yaml_config = load_config_from_yaml(args.config, args)
    
    main(args, yaml_config)

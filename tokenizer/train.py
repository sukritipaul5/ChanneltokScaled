#!/usr/bin/env python3
"""
Clean, minimal training script for ChannelTok tokenizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
import os
import argparse
import time
from tqdm import tqdm
import wandb
import gc

# Critical H100 optimizations - using new API to avoid deprecation warnings
torch.backends.cuda.matmul.fp32_precision = 'tf32'  # Use TF32 for matmul (H100 optimized)
torch.backends.cudnn.conv.fp32_precision = 'tf32'   # Use TF32 for convolutions (H100 optimized)
# Disable benchmarking to prevent CUDA workspace accumulation with dynamic masking
# Trades ~5-10% performance for memory stability when using variable execution paths
torch.backends.cudnn.benchmark = True
# Note: Don't use torch.set_float32_matmul_precision() - it conflicts with the above settings

# Import model and losses
from models import get_model_class
from models.discriminator import PatchGANDiscriminator
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import losses
from utils.checkpoint import load_config
from collections import defaultdict
from data import get_data_loader
from utils.lr_schedule import CosineDecayWithWarmupLRSchedule, WarmupLinearLRSchedule
from utils.helpers import upload_ckpt


class MinimalTrainer:
    """Minimal trainer without PyTorch Lightning overhead"""

    def __init__(self, config):
        self.config = config

        # Setup distributed FIRST to get correct device
        self.setup_distributed()

        # Initialize wandb (only on rank 0)
        self.setup_wandb()

        self.setup_model()

        # Setup mixed precision training
        self.setup_mixed_precision()

        # Setup losses (perceptual loss if configured)
        self.setup_losses()

        # Setup discriminator if GAN training is enabled
        self.setup_discriminator()

        # Create optimizer with fused AdamW
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.get('weight_decay', 0.01),
            fused=True  # Critical for H100
        )

        # Create discriminator optimizer if using GAN
        if self.use_gan:
            self.optimizer_d = torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=config.training.get('discriminator_lr', config.training.learning_rate),
                weight_decay=config.training.get('weight_decay', 0.01),
                fused=True
            )

        # Setup data loader
        self.train_loader = get_data_loader(
            config=self.config,
            device=self.device,
            distributed=self.distributed,
            local_rank=self.local_rank,
            world_size=self.world_size
        )

        # Force all ranks to wait until everyone has finished DALI setup
        if self.distributed:
            torch.distributed.barrier()
            if self.local_rank == 0:
                print("All ranks synchronized after data loader creation")

        # Calculate steps_per_epoch based on dataset type
        if self.config.data.dataset == 'imagefolder':
            # ImageFolder uses standard PyTorch DataLoader, so we can get length directly
            self.steps_per_epoch = len(self.train_loader)
            if self.local_rank == 0:
                total_samples = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') else "unknown"
                print(f"ImageFolder dataset: {total_samples} total samples, {self.steps_per_epoch} steps per epoch")
        elif self.config.data.dataset == 'imagenet_wds':
            # ImageNet-1k: 1,281,167 training images
            total_images = 1_281_167
            effective_batch = self.config.data.batch_size * self.world_size
            self.steps_per_epoch = total_images // effective_batch
            if self.local_rank == 0:
                print(f"Calculated steps_per_epoch for ImageNet: {self.steps_per_epoch} "
                      f"({total_images} images / {effective_batch} effective batch)")
        else:
            raise ValueError(f"Unknown dataset type: {self.config.data.dataset}")

        if self.local_rank == 0:
            print(f"Steps per epoch: {self.steps_per_epoch}")

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Setup learning rate scheduler
        self.lr_scheduler = self.setup_lr_scheduler()

        # Visualization settings
        self.log_viz_every = self.config.logging.get('log_viz_every', 5000)

        # Warmup CUDA kernels for both masked and unmasked paths
        #self.warmup_cuda_kernels()
        # if self.distributed:
        #     torch.distributed.barrier()
        # print("Warmup CUDA kernels complete")

    def setup_model(self):
        model_class = get_model_class(self.config.model.name)
        self.model = model_class(self.config)
        self.should_compile = bool(getattr(self.config.training, 'compile_model', False) and hasattr(torch, 'compile'))

        # Move model to device and wrap with DDP if distributed
        ddp_gradient_view = getattr(self.config.training, 'gradient_as_bucket_view', False)
        if self.distributed:
            self.model = self.model.to(self.device)
            if self.should_compile:
                try:
                    self.model = torch.compile(self.model)
                    if self.local_rank == 0:
                        print("Model compiled with torch.compile")
                except Exception as compile_error:
                    if self.local_rank == 0:
                        print(f"torch.compile failed ({compile_error}); continuing in eager mode")
                    self.should_compile = False

            for param in self.model.parameters():
                param.data = param.data.contiguous()

            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=ddp_gradient_view
            )
            if self.local_rank == 0:
                print(f"DDP initialized with gradient_as_bucket_view={ddp_gradient_view}")
            # if getattr(self.config.training, 'ensure_contiguous_grads', True):
            #     self._register_contiguous_gradients(self.model.module)

        else:
            self.model = self.model.to(self.device)
            if self.should_compile:
                try:
                    self.model = torch.compile(self.model)
                    print("Model compiled with torch.compile")
                except Exception as compile_error:
                    print(f"torch.compile failed ({compile_error}); continuing in eager mode")
                    self.should_compile = False



    def setup_lr_scheduler(self):
        """Setup learning rate scheduler based on config"""
        if not hasattr(self.config.training, 'lr_scheduler') or self.config.training.lr_scheduler is None:
            if self.local_rank == 0:
                print("No LR scheduler configured, using constant LR")
            return None

        sched_config = self.config.training.lr_scheduler
        sched_type = sched_config.get('type', 'cosine_warmup')

        # Calculate total training steps
        total_steps = self.config.training.max_epochs * self.steps_per_epoch

        if sched_type == 'cosine_warmup':
            scheduler = CosineDecayWithWarmupLRSchedule(
                optimizer=self.optimizer,
                init_lr=sched_config.get('init_lr', 1e-6),
                peak_lr=sched_config.get('peak_lr', self.config.training.learning_rate),
                min_lr=sched_config.get('min_lr', 1e-5),
                warmup_steps=sched_config.get('warmup_steps', 1000),
                total_steps=total_steps,
                current_step=self.global_step
            )
            if self.local_rank == 0:
                print(f"Using CosineDecayWithWarmupLRSchedule:")
                print(f"  init_lr={sched_config.get('init_lr', 1e-6)}, "
                      f"peak_lr={sched_config.get('peak_lr', self.config.training.learning_rate)}, "
                      f"min_lr={sched_config.get('min_lr', 1e-5)}")
                print(f"  warmup_steps={sched_config.get('warmup_steps', 1000)}, "
                      f"total_steps={total_steps}")

        elif sched_type == 'warmup_linear':
            scheduler = WarmupLinearLRSchedule(
                optimizer=self.optimizer,
                init_lr=sched_config.get('init_lr', 1e-6),
                peak_lr=sched_config.get('peak_lr', self.config.training.learning_rate),
                end_lr=sched_config.get('end_lr', 0.0),
                warmup_epochs=sched_config.get('warmup_steps', 1000),
                epochs=total_steps,
                current_step=self.global_step
            )
            if self.local_rank == 0:
                print(f"Using WarmupLinearLRSchedule with warmup_steps={sched_config.get('warmup_steps', 1000)}")

        else:
            raise ValueError(f"Unknown lr_scheduler type: {sched_type}")

        return scheduler

    def warmup_cuda_kernels(self):
        """Warmup both masked and unmasked paths to pre-compile CUDA kernels"""
        if not self.config.model.get('use_latent_mask', False):
            return

        if self.local_rank == 0:
            print("Warming up CUDA kernels for both execution paths...")

        self.model.train()
        dummy_input = torch.randn(
            2, self.config.model.image_channels,
            self.config.data.img_size, self.config.data.img_size,
            device=self.device
        )

        with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            with torch.no_grad():
                # Warmup unmasked path
                no_mask = torch.zeros((dummy_input.shape[0], 1, 1, 1), device=self.device, dtype=dummy_input.dtype)
                full_mask = torch.ones_like(no_mask)
                _ = self.model(dummy_input, mask_toggle=no_mask)
                # Warmup masked path
                _ = self.model(dummy_input, mask_toggle=full_mask)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        if self.local_rank == 0:
            print("CUDA kernel warmup complete")

    def _prepare_mask_toggle(self, images):
        """Create per-sample mask toggle tensor for latent masking."""
        if not self.config.model.get('use_latent_mask', False):
            return None

        warmup_epochs = self.config.model.get('latent_mask_warmup_epochs', 0)
        if self.epoch < warmup_epochs:
            mask_prob = 0.0
        else:
            mask_prob = float(self.config.model.get('latent_mask_prob', 0.5))

        mask_shape = (images.shape[0], 1, 1, 1)
        if mask_prob <= 0.0:
            toggle = images.new_zeros(mask_shape)
        elif mask_prob >= 1.0:
            toggle = images.new_ones(mask_shape)
        else:
            prob_tensor = images.new_full(mask_shape, mask_prob)
            toggle = torch.bernoulli(prob_tensor)

        return toggle.detach()

    def setup_distributed(self):
        """Setup DDP if running on multiple GPUs"""
        # Check if we're in a distributed environment
        self.distributed = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            # Initialize process group if not already initialized
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend='nccl')

            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.rank = int(os.environ.get('RANK', 0))  # Global rank across all nodes
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))

            # Set the device for this process
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')

            if self.rank == 0:
                print(f"Distributed training with {self.world_size} GPUs")
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_wandb(self):
        """Initialize wandb logging (only on global rank 0)"""
        # Check if wandb is explicitly disabled in config
        use_wandb = self.config.logging.get('use_wandb', True)

        if self.rank == 0 and use_wandb:
            wandb_id_path = os.path.join(self.config.output_dir, "wandb_id.txt")
            resume_id = None
            if os.path.exists(wandb_id_path):
                with open(wandb_id_path, "r") as f:
                    resume_id = f.read().strip() or None
            wandb.init(
                project=self.config.logging.get('project', 'flextok'),
                name=self.config.logging.get('name', 'clean_train'),
                tags=self.config.logging.get('tags', ['channeltok']),
                config=OmegaConf.to_container(self.config, resolve=True),
                dir=self.config.get('output_dir', './outputs'),
                resume='allow',
                id=resume_id
            )
            if not resume_id:
                with open(wandb_id_path, "w") as f:
                    f.write(wandb.run.id)
            self.use_wandb = True
            print("WandB initialized")
        else:
            self.use_wandb = False
            if self.rank == 0:
                print("WandB disabled (use_wandb=False in config)")

    def setup_mixed_precision(self):
        """Setup automatic mixed precision training"""
        precision = self.config.training.get('precision', '32')

        # Parse precision string (similar to PyTorch Lightning)
        if precision in ['16', '16-mixed', 'fp16']:
            self.use_amp = True
            self.amp_dtype = torch.float16
            print("Using FP16 mixed precision training")
        elif precision in ['bf16', 'bf16-mixed', 'bfloat16']:
            self.use_amp = True
            self.amp_dtype = torch.bfloat16
            print("Using BF16 mixed precision training (better for H100)")
        else:
            self.use_amp = False
            self.amp_dtype = torch.float32
            print("Using FP32 full precision training")

        # Create GradScaler for mixed precision (only for fp16, not bf16)
        if self.use_amp:
            if self.amp_dtype == torch.float16:
                self.scaler = torch.cuda.amp.GradScaler()
                print("Initialized GradScaler for FP16")
            else:
                self.scaler = None  # BF16 doesn't need loss scaling
                print("BF16 mode - no GradScaler needed")
        else:
            self.scaler = None

    def setup_losses(self):
        """Setup loss functions based on config - flexible system"""
        # Get loss configuration from config
        self.loss_list = self.config.loss.get('loss_list', ['l2', 'perceptual'])
        self.loss_weights = self.config.loss.get('loss_weights', [1.0, 0.1])

        # Build loss functions dictionary
        self.loss_functions = defaultdict(dict)
        for loss_name in self.loss_list:
            if loss_name not in losses.LOSS_REGISTRY:
                print(f"Warning: Loss '{loss_name}' not found in registry, skipping")
                continue

            info = {
                'name': loss_name,
                'weight': self.loss_weights[self.loss_list.index(loss_name)],
                'function': losses.LOSS_REGISTRY[loss_name]
            }
            self.loss_functions[loss_name] = info

        # Setup perceptual loss model if needed
        if 'perceptual' in self.loss_functions or 'lpips' in self.loss_functions:
            self.perceptual_loss_fn = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg',
                normalize=False
            ).to(self.device)
            self.perceptual_loss_fn.eval()
            for param in self.perceptual_loss_fn.parameters():
                param.requires_grad_(False)
        else:
            self.perceptual_loss_fn = None

        print(f"Initialized losses: {list(self.loss_functions.keys())}")
        print(f"Loss weights: {[info['weight'] for info in self.loss_functions.values()]}")

    def setup_discriminator(self):
        """Setup discriminator for GAN training if enabled"""
        self.use_gan = self.config.loss.get('use_gan', False)
        self.gan_start_epoch = self.config.loss.get('gan_start_epoch', 0)
        self.gan_loss_weight = self.config.loss.get('gan_loss_weight', 0.1)

        if self.use_gan:
            disc_channels = self.config.loss.get('discriminator_channels', 64)
            disc_layers = self.config.loss.get('discriminator_layers', 3)

            self.discriminator = PatchGANDiscriminator(
                input_channels=self.config.model.image_channels,
                base_channels=disc_channels,
                n_layers=disc_layers
            ).to(self.device)

            if self.distributed:
                self.discriminator = nn.parallel.DistributedDataParallel(
                    self.discriminator,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False
                )

            if self.local_rank == 0:
                print(f"Discriminator initialized (start_epoch={self.gan_start_epoch}, weight={self.gan_loss_weight})")
        else:
            self.discriminator = None

    def compute_loss(self, images, reconstructed):
        """Compute reconstruction losses (excludes adversarial loss)"""
        # Ensure both are in [-1, 1] range (non-inplace)
        reconstructed_fp32 = reconstructed.float().clamp(-1.0, 1.0)
        images_fp32 = images.float().clamp(-1.0, 1.0)

        loss_values = {}
        total_loss = 0.0

        # Pre-compute [0,1] versions once to avoid duplication
        reconstructed_01 = (reconstructed_fp32 + 1.0) * 0.5
        images_01 = (images_fp32 + 1.0) * 0.5

        # Build kwargs for loss functions
        loss_kwargs = {
            'perceptual_fn': self.perceptual_loss_fn,
        }

        # Compute each loss from the registry (excludes adversarial)
        for loss_name, loss_info in self.loss_functions.items():
            if 'perceptual' in loss_name or 'lpips' in loss_name:
                # Use [-1, 1] range for perceptual losses
                loss_val = loss_info['function'](
                    reconstructed_fp32,
                    images_fp32,
                    **loss_kwargs
                )
            else:
                # Use [0, 1] range for pixel-based losses
                loss_val = loss_info['function'](reconstructed_01, images_01, **loss_kwargs)

            # Apply weight and accumulate
            weighted_loss = loss_val * loss_info['weight']
            loss_values[loss_name] = loss_val.item()
            loss_values[f'{loss_name}_weight'] = loss_info['weight']
            total_loss += weighted_loss

            # Clean up immediately
            del loss_val, weighted_loss

        # Calculate PSNR for logging (reuse reconstructed_01, images_01)
        mse = F.mse_loss(reconstructed_01, images_01)
        psnr = 10 * torch.log10(1 / (mse + 1e-8))
        loss_values['psnr'] = psnr.item()

        # Clean up intermediate tensors before returning
        del reconstructed_fp32, images_fp32, reconstructed_01, images_01, mse, psnr

        return total_loss, loss_values

    def train_epoch(self):
        """Train one epoch - minimal overhead"""
        self.model.train()

        epoch_loss = 0.0
        epoch_psnr_sum = 0.0
        epoch_start = time.time()
        batch_start_time = time.time()
        data_time_sum = 0.0
        compute_time_sum = 0.0

        # Use tqdm for progress only (only on rank 0)
        if self.local_rank == 0:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}", ncols=120)
        else:
            pbar = self.train_loader

        fetch_start = time.time()

        for batch_idx, batch in enumerate(pbar):
            data_time = time.time() - fetch_start
            # Get images from batch
            # DALI returns list of dicts: [{'data': tensor}]
            # ImageFolder returns tuple: (images, labels)
            if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
                images = batch[0]['data']  # DALI format
            elif isinstance(batch, (list, tuple)):
                images = batch[0]  # ImageFolder format
            else:
                images = batch

            if not images.is_cuda:
                images = images.to(self.device,non_blocking=True)

            step_start = time.time()

            mask_toggle = self._prepare_mask_toggle(images)


            # Mixed precision context
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):

                # Forward pass
                #mask will be applied only if use_latent_mask is true in config.
                output = self.model(images, mask_toggle=mask_toggle)

                # Handle different output formats and extract quantizer aux loss
                quantizer_aux_loss = 0.0
                if isinstance(output, tuple):
                    reconstructed = output[0]
                    # Extract quantizer auxiliary loss if present (for LFQ/BSQ)
                    if isinstance(output[1], dict) and 'quantizer_aux_loss' in output[1]:
                        quantizer_aux_loss = output[1]['quantizer_aux_loss']
                else:
                    reconstructed = output

            # Train discriminator FIRST if GAN is enabled (standard GAN order: D then G)
            d_loss_val = 0.0
            if self.use_gan and self.epoch >= self.gan_start_epoch:
                self.optimizer_d.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    logits_real = self.discriminator(images.detach())
                    logits_fake = self.discriminator(reconstructed.detach())
                    d_loss = losses.hinge_d_loss(logits_real, logits_fake)
                    d_loss_val = d_loss.item()

                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(self.optimizer_d)
                    # Note: scaler.update() called once at end of iteration
                else:
                    d_loss.backward()
                    self.optimizer_d.step()

                del d_loss, logits_real, logits_fake

            # THEN compute generator loss
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                # Compute reconstruction losses (NO adversarial)
                loss, loss_dict = self.compute_loss(images, reconstructed)

                # Add quantizer auxiliary loss (entropy + commitment loss for LFQ/BSQ)
                loss = loss + quantizer_aux_loss

                # Add GAN loss manually if enabled
                g_loss_val = 0.0
                if self.use_gan and self.epoch >= self.gan_start_epoch:
                    logits_fake_g = self.discriminator(reconstructed)
                    g_loss = losses.hinge_g_loss(logits_fake_g)
                    loss = loss + self.gan_loss_weight * g_loss
                    g_loss_val = g_loss.item()
                    del logits_fake_g, g_loss

            # Backward pass with mixed precision
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.scaler is not None:
                # FP16 with gradient scaling
                self.scaler.scale(loss).backward()

                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                # Update scaler once per iteration (after both D and G)
                self.scaler.update()
            else:
                # FP32 or BF16 (no scaling needed)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Update learning rate if scheduler is configured
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Cleanup output tuple immediately (but keep reconstructed for potential viz logging)
            del output

            compute_time = time.time() - step_start
            data_time_sum += data_time
            compute_time_sum += compute_time

            # Update metrics
            epoch_loss += loss.item()
            epoch_psnr_sum += loss_dict['psnr']
            self.global_step += 1

            # Update progress bar (only on rank 0)
            avg_loss = epoch_loss / (batch_idx + 1)
            if self.local_rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'psnr': f'{loss_dict["psnr"]:.2f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.1e}',
                    'data_ms': f'{data_time * 1000:.1f}',
                    'comp_ms': f'{compute_time * 1000:.1f}'
                })

            # Log to wandb (only on global rank 0)
            if self.use_wandb and self.global_step % self.config.logging.get('log_every_n_steps', 50) == 0 and self.rank == 0:
                batch_time = time.time() - batch_start_time
                samples_per_sec = self.config.data.batch_size / batch_time if batch_time > 0 else 0

                log_dict = {
                    'train/total_loss': loss.item(),
                    'train/psnr': loss_dict['psnr'],
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/samples_per_sec': samples_per_sec,
                    'train/batch_time_ms': batch_time * 1000,
                    'train/data_time_ms': data_time * 1000,
                    'train/compute_time_ms': compute_time * 1000,
                    'train/global_step': self.global_step,
                }

                # Add all individual losses dynamically
                for loss_name in self.loss_functions.keys():
                    if loss_name in loss_dict:
                        log_dict[f'train/{loss_name}'] = loss_dict[loss_name]
                        log_dict[f'train/{loss_name}_weight'] = loss_dict.get(f'{loss_name}_weight', 0.0)

                # Log quantizer auxiliary loss (for LFQ/BSQ)
                if isinstance(quantizer_aux_loss, torch.Tensor):
                    log_dict['train/quantizer_aux_loss'] = quantizer_aux_loss.item()
                elif quantizer_aux_loss != 0.0:
                    log_dict['train/quantizer_aux_loss'] = quantizer_aux_loss

                # Log GAN losses if enabled
                if self.use_gan and self.epoch >= self.gan_start_epoch:
                    log_dict['train/d_loss'] = d_loss_val
                    log_dict['train/g_loss'] = g_loss_val

                wandb.log(log_dict, step=self.global_step)

                batch_start_time = time.time()

            # Visualization logging (minimal, only at specified intervals)
            if self.use_wandb and self.global_step % self.log_viz_every == 0 and self.rank == 0:
                self.log_reconstruction(images, reconstructed)

            # Delete large tensors after all uses
            del reconstructed, images, loss, loss_dict

            # # Aggressive memory cleanup every step (testing)
            # if self.global_step % 10 == 0:
            #     gc.collect()
            #     torch.cuda.empty_cache()

            # Memory monitoring and cleanup
            if self.global_step % 200 == 0:
                if self.local_rank == 0:
                    allocated_gb = torch.cuda.memory_allocated() / 1e9
                    reserved_gb = torch.cuda.memory_reserved() / 1e9
                    max_allocated_gb = torch.cuda.max_memory_allocated() / 1e9

                    # Also get Python process memory
                    import psutil
                    process = psutil.Process()
                    process_mem_gb = process.memory_info().rss / 1e9

                    print(f"[Step {self.global_step}] CUDA Memory: {allocated_gb:.2f}GB allocated, "
                          f"{reserved_gb:.2f}GB reserved, {max_allocated_gb:.2f}GB peak | "
                          f"Process RSS: {process_mem_gb:.2f}GB")

                    # Log to wandb if enabled
                    if self.use_wandb:
                        wandb.log({
                            'memory/allocated_gb': allocated_gb,
                            'memory/reserved_gb': reserved_gb,
                            'memory/peak_allocated_gb': max_allocated_gb,
                            'memory/process_rss_gb': process_mem_gb,
                        }, step=self.global_step)


            fetch_start = time.time()

        # Epoch summary with DDP-aware averaging
        epoch_time = time.time() - epoch_start

        # All-reduce epoch metrics across all ranks for true global average
        if self.distributed:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=self.device)
            epoch_psnr_tensor = torch.tensor(epoch_psnr_sum, device=self.device)

            torch.distributed.all_reduce(epoch_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(epoch_psnr_tensor, op=torch.distributed.ReduceOp.SUM)

            epoch_loss = epoch_loss_tensor.item()
            epoch_psnr_sum = epoch_psnr_tensor.item()

        # Compute true averages across all data (all ranks)
        total_batches = self.steps_per_epoch * (self.world_size if self.distributed else 1)
        avg_loss = epoch_loss / total_batches
        avg_psnr = epoch_psnr_sum / total_batches
        samples_per_sec = len(self.train_loader) * self.config.data.batch_size / epoch_time

        if self.local_rank == 0:
            print(f"\nEpoch {self.epoch} completed in {epoch_time:.1f}s")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Avg PSNR: {avg_psnr:.2f}")
            #print(f"  Throughput: {samples_per_sec:.1f} samples/sec")
            print(f"  Steps/sec: {len(self.train_loader)/epoch_time:.2f}")
            print(f"  Avg data time: {(data_time_sum / len(self.train_loader)) * 1000:.1f} ms")
            print(f"  Avg compute time: {(compute_time_sum / len(self.train_loader)) * 1000:.1f} ms")

            # Log epoch averages to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch/avg_loss': avg_loss,
                    'epoch/avg_psnr': avg_psnr,
                    'epoch/number': self.epoch,
                    'epoch/time_seconds': epoch_time,
                }, step=self.global_step)

        return avg_loss

    def log_reconstruction(self, images, reconstructed):
        """Log reconstruction visualization to wandb"""
        with torch.no_grad():
            # Take first 8 images for visualization
            n_viz = min(8, images.shape[0])
            images_viz = images[:n_viz].cpu()
            reconstructed_viz = reconstructed[:n_viz].cpu()

            # Convert from [-1, 1] to [0, 1]
            images_viz = (images_viz + 1.0) / 2.0
            reconstructed_viz = (reconstructed_viz + 1.0) / 2.0

            # Clamp values
            images_viz = images_viz.clamp(0, 1)
            reconstructed_viz = reconstructed_viz.clamp(0, 1)

            # Create grid of original and reconstructed images
            import torchvision.utils as vutils
            combined = torch.cat([images_viz, reconstructed_viz], dim=0)
            grid = vutils.make_grid(combined, nrow=n_viz, normalize=False, scale_each=False)

            # Log to wandb
            wandb.log({
                'visualizations/reconstruction': wandb.Image(
                    grid.permute(1, 2, 0).numpy(),
                    caption=f"Top: Original, Bottom: Reconstructed (Step {self.global_step})"
                )
            }, step=self.global_step)

    def save_checkpoint(self, path):
        """Save minimal checkpoint"""
        if self.local_rank != 0:
            return
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if self.distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': OmegaConf.to_container(self.config),
            'avg_loss': getattr(self, '_last_epoch_loss', None)
        }
        if self.use_gan:
            checkpoint['discriminator_state_dict'] = self.discriminator.module.state_dict() if self.distributed else self.discriminator.state_dict()
            checkpoint['optimizer_d_state_dict'] = self.optimizer_d.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def _move_optimizer_state_to_device(self, optimizer):
        """Ensure optimizer state tensors live on the correct device."""
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def load_checkpoint(self, path):
        if self.distributed:
            checkpoint_container = [None]
            if self.rank == 0:
                checkpoint_container[0] = torch.load(path, map_location='cpu', weights_only=False)
            torch.distributed.broadcast_object_list(checkpoint_container, src=0)
            checkpoint = checkpoint_container[0]
        else:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = checkpoint['model_state_dict']
        target_model = self.model.module if self.distributed else self.model
        target_model.load_state_dict(state_dict, strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._move_optimizer_state_to_device(self.optimizer)

        if self.use_gan and 'discriminator_state_dict' in checkpoint:
            target_disc = self.discriminator.module if self.distributed else self.discriminator
            target_disc.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            self._move_optimizer_state_to_device(self.optimizer_d)
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self._last_epoch_loss = checkpoint.get('avg_loss', None)

        # Recreate LR scheduler with restored global_step to continue from correct position
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.setup_lr_scheduler()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Resumed from {path} (epoch {self.epoch}, step {self.global_step})")
            print(f"LR scheduler reinitialized: current_lr={current_lr:.2e}")
        else:
            print(f"Resumed from {path} (epoch {self.epoch}, step {self.global_step})")
        if self.distributed:
            torch.distributed.barrier()

    def count_parameters(self, model):
        """Count trainable parameters in model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def load_initial_weights(self, path):
        """Load pretrained generator/discriminator weights without optimizer state."""
        if self.distributed:
            checkpoint_container = [None]
            if self.rank == 0:
                checkpoint_container[0] = torch.load(path, map_location='cpu', weights_only=False)
            torch.distributed.broadcast_object_list(checkpoint_container, src=0)
            checkpoint = checkpoint_container[0]
        else:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        target_model = self.model.module if self.distributed else self.model
        target_model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if self.use_gan and 'discriminator_state_dict' in checkpoint:
            target_disc = self.discriminator.module if self.distributed else self.discriminator
            target_disc.load_state_dict(checkpoint['discriminator_state_dict'], strict=True)

        if self.rank == 0:
            print(f"Initialized model weights from {path}")
        if self.distributed:
            torch.distributed.barrier()

    def train(self):
        """Main training loop - no unnecessary operations"""
        print("\nStarting training...")
        print(f"Total epochs: {self.config.training.max_epochs}")
        print(f"Batch size: {self.config.data.batch_size}")
        print(f"Learning rate: {self.config.training.learning_rate}")

        # Print model parameters (only on rank 0)
        if self.local_rank == 0:
            model_params = self.count_parameters(self.model)
            print(f"Generator parameters: {model_params / 1e6:.2f}M")

            if self.use_gan:
                disc_params = self.count_parameters(self.discriminator)
                print(f"Discriminator parameters: {disc_params / 1e6:.2f}M")
                print(f"Total parameters: {(model_params + disc_params) / 1e6:.2f}M")

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_dir = os.path.join(self.config.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save config as reference (only on rank 0)
        if self.rank == 0:
            config_save_path = os.path.join(checkpoint_dir, 'config.yaml')
            if not os.path.exists(config_save_path):
                OmegaConf.save(self.config, config_save_path)
                print(f"Saved config to {config_save_path}")
        best_ckpt_path = os.path.join(checkpoint_dir, 'best.ckpt')
        last_ckpt_path = os.path.join(checkpoint_dir, 'last.ckpt')
        resume_ckpt = None
        init_weights_path = self.config.get('init_weights')
        if isinstance(init_weights_path, str):
            init_weights_path = os.path.expanduser(os.path.expandvars(init_weights_path))
        else:
            init_weights_path = None
        if self.rank == 0:
            if os.path.exists(last_ckpt_path) and self.config.get('resume', False):
                resume_ckpt = last_ckpt_path
            elif isinstance(self.config.get('resume_from'), str) and os.path.exists(self.config.resume_from):
                resume_ckpt = self.config.resume_from

        if self.distributed:
            resume_container = [resume_ckpt]
            torch.distributed.broadcast_object_list(resume_container, src=0)
            resume_ckpt = resume_container[0]

        if resume_ckpt:
            self.load_checkpoint(resume_ckpt)
            start_epoch = self.epoch + 1
            if os.path.exists(best_ckpt_path):
                best_loss = torch.load(best_ckpt_path, map_location='cpu').get('avg_loss', float('inf'))
            else:
                best_loss = float('inf')
            if self.rank == 0:
                print(f"Resuming training from epoch {start_epoch}")
        else:
            if init_weights_path:
                if not os.path.exists(init_weights_path):
                    raise FileNotFoundError(f"init_weights path does not exist: {init_weights_path}")
                self.load_initial_weights(init_weights_path)
            best_loss = float('inf')
            start_epoch = 0

        for epoch in range(start_epoch, self.config.training.max_epochs):
            self.epoch = epoch

            # Train epoch
            avg_loss = self.train_epoch()
            self._last_epoch_loss = avg_loss

            # Save last checkpoint every epoch (only on global rank 0)
            if self.rank == 0:
                self.save_checkpoint(last_ckpt_path)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                if self.rank == 0:
                    self.save_checkpoint(best_ckpt_path)

                    if self.epoch % 5 == 0:
                        upload_ckpt(self.config.output_dir)

            reset_epoch = int(self.config.training.get('reset_train_epoch', 0))
            if reset_epoch > 0 and (self.epoch + 1) % reset_epoch == 0:
                # Synchronize epoch count across all ranks before returning
                if self.distributed:
                    epoch_tensor = torch.tensor(self.epoch + 1, device=self.device)
                    torch.distributed.broadcast(epoch_tensor, src=0)
                    epoch_to_return = epoch_tensor.item()
                    torch.distributed.barrier()  # Ensure all ranks synced
                else:
                    epoch_to_return = self.epoch + 1

                if self.rank == 0:
                    print(f"Stopping after epoch {self.epoch} due to reset_train_epoch={reset_epoch}")
                return epoch_to_return

        print("\nTraining completed!")
        if self.rank == 0:
            upload_ckpt(self.config.output_dir)
        # Close wandb
        if self.use_wandb:
            wandb.finish()

        # Synchronize final epoch count across ranks
        if self.distributed:
            epoch_tensor = torch.tensor(self.config.training.max_epochs, device=self.device)
            torch.distributed.broadcast(epoch_tensor, src=0)
            return epoch_tensor.item()

        return self.config.training.max_epochs


def main():
    parser = argparse.ArgumentParser(description="Clean training script")
    parser.add_argument("-c", "--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()

    # Load config using shared utility
    base_config = load_config(args.config)

    reset_epochs = int(base_config.training.get('reset_train_epoch', 0))
    total_epochs = int(base_config.training.get('max_epochs', 0))
    if total_epochs <= 0:
        raise ValueError("config.training.max_epochs must be > 0")

    if reset_epochs <= 0:
        trainer = MinimalTrainer(base_config)
        trainer.train()
        return

    epochs_completed = 0
    while epochs_completed < total_epochs:
        # Create a fresh copy of the config for each trainer instance
        run_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        trainer = MinimalTrainer(run_config)
        epochs_completed = trainer.train()
        if epochs_completed >= total_epochs:
            break


if __name__ == "__main__":
    # Keep TF32 state consistent for torch.compile by setting both new and legacy APIs
    if hasattr(torch._C, "_set_allow_tf32"):
        torch._C._set_allow_tf32(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()

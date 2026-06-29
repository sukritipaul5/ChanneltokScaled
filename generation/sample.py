"""
Generation script for variable-length FSQ sequences with EOS token support
"""
import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
from tqdm import tqdm
from generation.models.gpt import GPT_models
from utils.checkpoint import load_config


@torch.no_grad()
def generate_variable_length(
    model, 
    batch_size=1,
    max_length=512,
    temperature=1.0,
    top_k=None,
    top_p=None,
    bos_token_id=63999,
    eos_token_id=63998,
    pad_token_id=64000,
    device='cuda'
):
    """
    Generate variable-length FSQ sequences that stop at EOS token.
    
    Args:
        model: The trained GPT model
        batch_size: Number of sequences to generate
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        device: Device to run on
    
    Returns:
        generated_sequences: List of generated sequences (without BOS/EOS)
        sequence_lengths: Actual lengths of generated sequences
    """
    model.eval()
    
    # Start with BOS token for each sequence in batch
    generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # For unconditional generation, create dummy conditioning
    dummy_cond = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for step in range(max_length - 1):  # -1 because we already have BOS
        # Get model predictions
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = model(idx=generated, cond_idx=dummy_cond, targets=None)
        
        # Get logits for the last position
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('inf')
        
        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
        
        # For finished sequences, replace with padding token
        next_token[finished] = pad_token_id
        
        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)
        
        # Check if any sequence generated EOS token
        finished = finished | (next_token.squeeze(1) == eos_token_id)
        
        # If all sequences are finished, stop
        if finished.all():
            break
    
    # Process generated sequences to remove BOS, EOS, and padding
    generated_sequences = []
    sequence_lengths = []
    
    for seq in generated:
        # Find EOS position (if exists)
        eos_positions = (seq == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            eos_pos = eos_positions[0].item()
        else:
            eos_pos = len(seq)
        
        # Extract sequence without BOS and EOS
        # Skip first token (BOS) and stop before EOS
        clean_seq = seq[1:eos_pos]
        
        generated_sequences.append(clean_seq.cpu().numpy())
        sequence_lengths.append(len(clean_seq))
    
    return generated_sequences, sequence_lengths


def main():
    parser = argparse.ArgumentParser(description='Generate variable-length FSQ sequences')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for generation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=None, help='Top-p (nucleus) sampling')
    parser.add_argument('--output-dir', type=str, default='generated_samples', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading model {config.gpt_model}...")
    
    # Model needs vocab_size + 1 for padding token
    model_vocab_size = config.vocab_size + 1
    block_size = config.max_sequence_length
    
    model = GPT_models[config.gpt_model](
        vocab_size=model_vocab_size,
        block_size=block_size,
        num_classes=config.num_classes,
        cls_token_num=config.cls_token_num,
        model_type=config.gpt_type,
        resid_dropout_p=0.0,  # No dropout for inference
        ffn_dropout_p=0.0,
        drop_path_rate=0.0,
        token_dropout_p=0.0,
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DDP)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    all_sequences = []
    all_lengths = []
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches)):
        current_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
        
        sequences, lengths = generate_variable_length(
            model,
            batch_size=current_batch_size,
            max_length=config.max_sequence_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
            device=device
        )
        
        all_sequences.extend(sequences)
        all_lengths.extend(lengths)
    
    # Save generated sequences
    output_file = os.path.join(args.output_dir, 'generated_sequences.npz')
    np.savez(
        output_file,
        sequences=np.array(all_sequences, dtype=object),
        lengths=np.array(all_lengths),
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length
    )
    
    print(f"Saved {len(all_sequences)} generated sequences to {output_file}")
    print(f"Sequence lengths - Min: {min(all_lengths)}, Max: {max(all_lengths)}, Mean: {np.mean(all_lengths):.1f}")
    
    # Print some statistics
    print("\nGeneration Statistics:")
    print(f"Total sequences generated: {len(all_sequences)}")
    print(f"Average sequence length: {np.mean(all_lengths):.1f}")
    print(f"Std sequence length: {np.std(all_lengths):.1f}")
    
    # Show length distribution
    length_hist, bins = np.histogram(all_lengths, bins=10)
    print("\nLength distribution:")
    for i in range(len(length_hist)):
        print(f"  {bins[i]:.0f}-{bins[i+1]:.0f}: {length_hist[i]} sequences")


if __name__ == "__main__":
    main()
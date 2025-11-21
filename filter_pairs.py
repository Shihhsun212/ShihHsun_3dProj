#!/usr/bin/env python3
"""
Filter image pairs using Doppelgangers++ classifier
To be placed in doppelgangers-plusplus directory
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import sys
import numpy as np
from scipy.special import softmax

# Correct imports based on the repository structure
from train import Doppelgangers
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs


def filter_pairs(
    image_dir: Path,
    pairs_file: Path,
    output_file: Path,
    checkpoint: Path,
    threshold: float = 0.8,
    device: str = 'cuda'
):
    """
    Filter image pairs using Doppelgangers++ classifier.
    
    Args:
        image_dir: Directory containing images
        pairs_file: Input pairs file (format: "img1.jpg img2.jpg" per line)
        output_file: Output filtered pairs file
        checkpoint: Path to Doppelgangers++ checkpoint (.pth file)
        threshold: Probability threshold (keep pairs with prob >= threshold)
        device: Device to run on ('cuda' or 'cpu')
    """
    
    print("="*70)
    print("Doppelgangers++ Pair Filtering")
    print("="*70)
    print(f"Image directory: {image_dir}")
    print(f"Input pairs:     {pairs_file}")
    print(f"Output pairs:    {output_file}")
    print(f"Checkpoint:      {checkpoint}")
    print(f"Threshold:       {threshold}")
    print("="*70)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model (using the same approach as colmap_usage.py)
    print(f"\nLoading Doppelgangers++ model...")
    try:
        model = AsymmetricMASt3R(
            pos_embed='RoPE100', 
            patch_embed_cls='ManyAR_PatchEmbed', 
            img_size=(512, 512), 
            head_type='catmlp+dpt', 
            head_type_dg='transformer',
            output_mode='pts3d+desc24', 
            output_mode_dg='dg_score', 
            depth_mode=('exp', -np.inf, np.inf), 
            conf_mode=('exp', 1, np.inf),
            enc_embed_dim=1024, 
            enc_depth=24, 
            enc_num_heads=16, 
            dec_embed_dim=768, 
            dec_depth=12, 
            dec_num_heads=12, 
            two_confs=True, 
            desc_conf_mode=('exp', 0, np.inf),
            add_dg_pred_head=True, 
            freeze=['mask','encoder','decoder','head']
        ).from_pretrained(str(checkpoint)).to(device)
        
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Read pairs
    print(f"\nReading pairs from {pairs_file}...")
    pairs = []
    try:
        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        print(f"✓ Loaded {len(pairs)} pairs")
    except Exception as e:
        print(f"✗ Error reading pairs: {e}")
        sys.exit(1)
    
    # Filter pairs
    print(f"\nFiltering pairs (threshold={threshold})...")
    filtered_pairs = []
    probabilities = []
    skipped = 0
    errors = 0
    
    with torch.no_grad():
        for img1_name, img2_name in tqdm(pairs, desc="Processing"):
            img1_path = image_dir / img1_name
            img2_path = image_dir / img2_name
            
            # Check if images exist
            if not img1_path.exists():
                if skipped < 5:  # Only print first 5
                    print(f"Warning: Image not found: {img1_path}")
                skipped += 1
                continue
            if not img2_path.exists():
                if skipped < 5:
                    print(f"Warning: Image not found: {img2_path}")
                skipped += 1
                continue
            
            try:
                # Load images using dust3r's load_images function
                img_paths = [str(img1_path), str(img2_path)]
                images = load_images(img_paths, size=512, verbose=False)
                
                # Run inference
                output = inference(make_pairs(images), model, device, verbose=False)
                
                # Extract predictions
                pred1 = output['pred1']
                pred2 = output['pred2']
                
                if isinstance(pred1, list):
                    pred1 = torch.stack(pred1, dim=0)
                if isinstance(pred2, list):
                    pred2 = torch.stack(pred2, dim=0)
                
                # Calculate scores using softmax (same as colmap_usage.py)
                from scipy.special import softmax
                score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
                score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)
                
                # Voting mechanism
                vote_0 = sum(score_s1[:,0] > score_s1[:,1]) + sum(score_s2[:,0] > score_s2[:,1])
                vote_1 = sum(score_s1[:,1] > score_s1[:,0]) + sum(score_s2[:,1] > score_s2[:,0])
                
                if vote_1 > vote_0:
                    score = np.max((score_s1[:,1], score_s2[:,1]))
                elif vote_1 < vote_0:
                    score = np.min((score_s1[:,1], score_s2[:,1]))
                else:
                    score = np.mean((score_s1[:,1], score_s2[:,1]))
                
                probabilities.append(score)
                
                # Keep pairs with score >= threshold (NOT doppelgangers)
                if score >= threshold:
                    filtered_pairs.append((img1_name, img2_name))
                    
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only print first 5 errors
                    print(f"\nError processing ({img1_name}, {img2_name}): {e}")
                continue
    
    # Save filtered pairs
    print(f"\nSaving filtered pairs to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_file, 'w') as f:
            for img1, img2 in filtered_pairs:
                f.write(f"{img1} {img2}\n")
        print("✓ Filtered pairs saved")
    except Exception as e:
        print(f"✗ Error saving pairs: {e}")
        sys.exit(1)
    
    # Print statistics
    print("\n" + "="*70)
    print("Filtering Results")
    print("="*70)
    print(f"Original pairs:  {len(pairs)}")
    print(f"Filtered pairs:  {len(filtered_pairs)}")
    print(f"Removed pairs:   {len(pairs) - len(filtered_pairs)} ({100*(len(pairs)-len(filtered_pairs))/len(pairs):.1f}%)")
    print(f"Skipped:         {skipped}")
    print(f"Errors:          {errors}")
    
    if probabilities:
        print(f"\nProbability Statistics:")
        print(f"  Mean:   {np.mean(probabilities):.3f}")
        print(f"  Median: {np.median(probabilities):.3f}")
        print(f"  Std:    {np.std(probabilities):.3f}")
        print(f"  Min:    {np.min(probabilities):.3f}")
        print(f"  Max:    {np.max(probabilities):.3f}")
    
    print("="*70)
    
    return filtered_pairs


def main():
    parser = argparse.ArgumentParser(
        description='Filter image pairs using Doppelgangers++ classifier'
    )
    parser.add_argument(
        '--images',
        type=Path,
        required=True,
        help='Directory containing images'
    )
    parser.add_argument(
        '--pairs',
        type=Path,
        required=True,
        help='Input pairs file (format: "img1.jpg img2.jpg" per line)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output filtered pairs file'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to Doppelgangers++ checkpoint (.pth file)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help='Probability threshold for filtering (default: 0.8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.images.exists():
        print(f"Error: Image directory does not exist: {args.images}")
        sys.exit(1)
    
    if not args.pairs.exists():
        print(f"Error: Pairs file does not exist: {args.pairs}")
        sys.exit(1)
    
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file does not exist: {args.checkpoint}")
        sys.exit(1)
    
    # Run filtering
    filter_pairs(
        args.images,
        args.pairs,
        args.output,
        args.checkpoint,
        args.threshold,
        args.device
    )


if __name__ == '__main__':
    main()

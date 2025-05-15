import torch
import random

def create_mask(batch_size, win_len, feature_size, mask_ratio=0.75):
    """
    Create a binary mask for masked modeling tasks.
    
    Args:
        batch_size (int): Batch size
        win_len (int): Window length (time dimension)
        feature_size (int): Feature dimension
        mask_ratio (float): Ratio of positions to mask (0.0-1.0)
        
    Returns:
        mask (torch.Tensor): Binary mask of shape [batch_size, 1, win_len, feature_size]
                            where 1 means keep, 0 means mask
    """
    # Get the number of patches
    num_patches = win_len * feature_size
    
    # Compute the number of patches to keep
    keep_patches = int(num_patches * (1 - mask_ratio))
    
    # Create the mask for one sample
    noise = torch.rand(batch_size, 1, win_len, feature_size)
    
    # Sort noise for each sample
    ids_shuffle = torch.argsort(noise.flatten(2), dim=2)
    ids_restore = torch.argsort(ids_shuffle, dim=2)
    
    # Generate binary mask
    mask = torch.zeros_like(noise.flatten(2))
    mask[:, :, :keep_patches] = 1
    
    # Unshuffle to get the mask
    mask = torch.gather(mask, dim=2, index=ids_restore)
    mask = mask.reshape(batch_size, 1, win_len, feature_size)
    
    return mask

def create_block_mask(batch_size, win_len, feature_size, block_size=8, mask_ratio=0.5):
    """
    Create a block-wise binary mask for masked modeling tasks.
    
    Args:
        batch_size (int): Batch size
        win_len (int): Window length (time dimension)
        feature_size (int): Feature dimension
        block_size (int): Size of mask blocks
        mask_ratio (float): Ratio of blocks to mask (0.0-1.0)
        
    Returns:
        mask (torch.Tensor): Binary mask of shape [batch_size, 1, win_len, feature_size]
                            where 1 means keep, 0 means mask
    """
    # Calculate number of blocks in each dimension
    time_blocks = win_len // block_size
    freq_blocks = feature_size // block_size
    
    # Ensure at least one block in each dimension
    time_blocks = max(1, time_blocks)
    freq_blocks = max(1, freq_blocks)
    
    # Create block-level mask
    block_mask = torch.ones(batch_size, 1, time_blocks, freq_blocks)
    
    # Number of blocks to mask
    num_blocks = time_blocks * freq_blocks
    blocks_to_mask = int(num_blocks * mask_ratio)
    
    # Randomly select blocks to mask for each sample
    for i in range(batch_size):
        # Flatten the block indices
        flat_indices = list(range(num_blocks))
        random.shuffle(flat_indices)
        
        # Select blocks to mask
        mask_indices = flat_indices[:blocks_to_mask]
        
        # Convert flat indices to 2D coordinates
        for idx in mask_indices:
            t_idx = idx // freq_blocks
            f_idx = idx % freq_blocks
            block_mask[i, 0, t_idx, f_idx] = 0
    
    # Upsample block mask to full resolution
    mask = torch.ones(batch_size, 1, win_len, feature_size)
    
    # Apply block mask
    for b in range(batch_size):
        for t in range(time_blocks):
            t_start = t * block_size
            t_end = min(win_len, (t + 1) * block_size)
            
            for f in range(freq_blocks):
                f_start = f * block_size
                f_end = min(feature_size, (f + 1) * block_size)
                
                mask[b, 0, t_start:t_end, f_start:f_end] = block_mask[b, 0, t, f]
    
    return mask

import torch

def create_mask(
    batch_size,
    seq_len=250,
    feature_size=98,
    row_mask_ratio=0.1,
    col_mask_ratio=0.1,
    device=None,
    dtype=torch.float32,
    seed=None
):
    """
    Creates a binary mask of shape (B, seq_len, feature_size), where a portion of
    rows and columns are masked out with 1s (rest are 0s).

    Args:
        batch_size (int): Number of samples
        seq_len (int): Length of sequence (e.g., time steps)
        feature_size (int): Number of features (e.g., frequency bins)
        row_mask_ratio (float): Proportion of rows to mask per sample
        col_mask_ratio (float): Proportion of columns to mask per sample
        device (torch.device): Device to place the tensor on
        dtype (torch.dtype): Type of the returned tensor
        seed (int): Optional random seed for reproducibility

    Returns:
        mask (torch.Tensor): Shape (B, seq_len, feature_size), with 1s as masked.
    """
    if seed is not None:
        torch.manual_seed(seed)

    row_mask = torch.zeros((batch_size, seq_len, feature_size), dtype=dtype, device=device)
    num_rows_to_mask = int(row_mask_ratio * seq_len)

    for i in range(batch_size):
        row_indices = torch.randperm(seq_len, device=device)[:num_rows_to_mask]
        row_mask[i, row_indices, :] = 1.0

    col_mask = torch.zeros((batch_size, seq_len, feature_size), dtype=dtype, device=device)
    num_cols_to_mask = int(col_mask_ratio * feature_size)

    for i in range(batch_size):
        col_indices = torch.randperm(feature_size, device=device)[:num_cols_to_mask]
        col_mask[i, :, col_indices] = 1.0

    return torch.maximum(row_mask, col_mask)

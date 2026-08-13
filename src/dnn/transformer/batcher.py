"""
Simple batching function.
"""
import torch.stack, torch.randint


def get_batch(data, block_size, batch_size, device):
    """
    Randomly sample contiguous batch_size chunks of length 'block_size'
    from data, and for each chunk poroduces a matching target sequence
    shifted by one token.

    To Consider - data[i:i+block_size] on a large tensor already in memory
    is fine, but if data is memory-mapped (e.g. np.memmap), it is typically
    converted with torch.from_numpy(...).long() inside this function to 
    avoid holding the whole file in RAM — worth checking 
    you hit memory issues.
    """
    # Pick batch_size random starting indices into 1D tensor of token.
    # For example: If data = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21], block_size = 2, batch_size = 5, then ix might be [0, 4, 7] and batch_size = 5
    # high = 11 - 2 = 9, so ix is sampled from [0, 9) and could be [0, 3, 5, 8, 2] for example.
    high = len(data) - block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Build input batch: for each random start index i, grab a contiguous
    # chunk of length block_size (context window. Stacking gives
    # shape (batch_size, block_size)
    # From above example, x = tensor [[ 1, 3], [ 7, 9], [11, 13], [17, 19], [5, 7]]
    x = torch.stack([data[i : i + block_size] for i in ix])

    # Builds the target batch, the same chunks but shifted one position
    # to the right. For predicting the next token.
    # From above example, y = tensor [[ 3, 5], [ 9, 11], [13, 15], [19, 21], [7, 9]]
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    # Move both tensors to cpu/gpu
    return x.to(device), y.to(device)
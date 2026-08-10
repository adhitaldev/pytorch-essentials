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
    # Pick batch_size random starting indices into 1D tensor of 
    # token. 
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Build input batch: for each random start i, grab a contiguous
    # chunk of length block_size (context window. Stacking gives
    # shape (batch_size, block_size)

    x = torch.stack([data[i:i + block_size] for i in ix])

    # Builds the target batch, the same chunks but shifted one position
    # to the right. For predicting the next token.
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])

    # Move both tensors to cpu/gpu
    return x.to(device), y.to(device)
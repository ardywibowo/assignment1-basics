import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    xs = np.stack([dataset[s:s+context_length] for s in starts])
    ys = np.stack([dataset[s+1:s+context_length+1] for s in starts])
    x_tensor = torch.tensor(xs, dtype=torch.long, device=device)
    y_tensor = torch.tensor(ys, dtype=torch.long, device=device)
    return x_tensor, y_tensor

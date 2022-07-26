import torch


def get_device(device):
    """Select a backend from CPU, CUDA, or MPS."""
    devices = ["cpu", "cuda", "mps"]
    if device == "cpu":
        return torch.device("cpu")
    elif device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("cuda is not available")
    elif device == "mps":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError("mps is not available")
    else:
        raise ValueError(f"device must be one of {devices}")

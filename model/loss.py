import torch.nn.functional as F
import torch


def nll_loss(output: torch.tensor, target: torch.tensor, **kwargs):
    """
    A softmax activation is required at the final layer of the network
    """
    return F.nll_loss(output, target, **kwargs)


def cse(output: torch.tensor, target: torch.tensor, **kwargs):
    """
    With pytorch cross entropy loss a softmax activation is NOT required
    at the final layer of network.
    """
    return F.cross_entropy(output, target, **kwargs)


def mse(output: torch.tensor, target: torch.tensor, **kwargs):
    return F.mse_loss(output, target, **kwargs)

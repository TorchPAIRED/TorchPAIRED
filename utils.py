from torch import nn
import torch
from torch import distributions, nn

def make_n_hidden_layers(n, size):
    hiddens = []
    for i in range(n):
        hiddens.append(nn.Linear(size, size))
        hiddens.append(nn.ReLU())
    return hiddens

def get_squashed_diagonal_gaussian_head_fun(action_size):
    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = torch.chunk(x, 2, dim=1)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = distributions.Independent(
            distributions.Normal(loc=mean, scale=torch.sqrt(var)), 1
        )
        # cache_size=1 is required for numerical stability
        return distributions.transformed_distribution.TransformedDistribution(
            base_distribution, [distributions.transforms.TanhTransform(cache_size=1)]
        )
    return squashed_diagonal_gaussian_head
from flow import Flow, GaussianBase, MaskedCouplingLayer
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians prior with learnable parameters.

        Parameters:
        M: [int]
           Dimension of the latent space.
        K: [int]
           Number of mixture components.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K

        # Learnable mixture logits, means, and log-variances
        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(torch.randn(K, M))
        self.log_stds = nn.Parameter(torch.zeros(K, M))

    def forward(self):
        """
        Return the MoG prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        mixture = td.Categorical(logits=self.logits)
        components = td.Independent(
            td.Normal(loc=self.means, scale=torch.exp(self.log_stds)),
            1,
        )
        return td.MixtureSameFamily(mixture, components)


# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

from flow import Flow, GaussianBase, MaskedCouplingLayer
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from MoGPrior import MoGPrior
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class FlowPrior(nn.Module):
    def __init__(self, flow):
        """
        Wrap a Flow model as a prior distribution for the VAE.
        
        Parameters:
        flow: [Flow]
            A trained (or jointly trained) Flow model.
        """
        super(FlowPrior, self).__init__()
        self.flow = flow

    def forward(self):
        return self

    def log_prob(self, z):
        return self.flow.log_prob(z)

    def sample(self, sample_shape=torch.Size()):
        n = sample_shape[0] if len(sample_shape) > 0 else 1
        return self.flow.sample((n,))

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        elbo = torch.mean(self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z), dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


    # model.eval()
    # with torch.no_grad():
    #     # Sample from prior
    #     z_prior = model.prior().sample(torch.Size([10000])).cpu().numpy()

    #     # Get aggregated posterior samples by encoding data
    #     z_posterior = []
    #     labels = []
    #     for x, y in mnist_test_loader:
    #         x = x.to(device)
    #         q = model.encoder(x)
    #         z_posterior.append(q.rsample().cpu().numpy())
    #         labels.append(y.numpy())
    #     z_posterior = np.concatenate(z_posterior, axis=0)[:10000]
    #     labels = np.concatenate(labels, axis=0)[:10000]

    #     # Fit PCA on combined data
    #     combined = np.concatenate([z_prior, z_posterior], axis=0)
    #     pca = PCA(n_components=2)
    #     pca.fit(combined)

    #     z_prior_pca = pca.transform(z_prior)
    #     z_posterior_pca = pca.transform(z_posterior)

    # fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # # Prior plot
    # axes[0].scatter(z_prior_pca[:, 0], z_prior_pca[:, 1], alpha=0.3, s=5, c="steelblue")
    # axes[0].set_title("Prior")
    # axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    # axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")

    # # Posterior plot colored by digit class
    # cmap = plt.get_cmap("tab10")
    # for digit in range(10):
    #     mask = labels == digit
    #     axes[1].scatter(z_posterior_pca[mask, 0], z_posterior_pca[mask, 1],
    #                     alpha=0.3, s=5, c=[cmap(digit)], label=str(digit))
    # axes[1].set_title("Aggregated Posterior (colored by digit)")
    # axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    # axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    # axes[1].legend(title="Digit", markerscale=3, loc="best")

    # plt.suptitle(f"Prior vs Aggregated Posterior (PCA)\nExplained variance: {pca.explained_variance_ratio_.sum():.2%}")
    # plt.tight_layout()
    # plt.savefig("prior_vs_posterior_pca.png")
    # plt.show()


def evalELBO(model, test_loader, device):
    total_elbo = 0.0
    num_batches = 0
    with torch.no_grad():
        for x, _ in test_loader:
            elbo = model.elbo(x.to(device))
            total_elbo += elbo.item()
            num_batches += 1
    print(f"Average ELBO: {total_elbo / num_batches:.4f}")


def train_and_eval_multiple_runs(model, optimizer, train_loader, test_loader, epochs, device, num_runs=10):
    """
    Train the model multiple times and evaluate the ELBO for each trained model.

    Parameters:
    model: [VAE]
        The VAE model to train and evaluate.
    optimizer: [torch.optim.Optimizer]
        The optimizer to use for training.
    train_loader: [torch.utils.data.DataLoader]
        The data loader for the training set.
    test_loader: [torch.utils.data.DataLoader]
        The data loader for the test set.
    epochs: [int]
        Number of epochs to train for each run.
    device: [torch.device]
        The device to use for training and evaluation.
    num_runs: [int]
        Number of training and evaluation runs.

    Returns:
    mean_elbo: [float]
        The mean ELBO over the runs.
    std_elbo: [float]
        The standard deviation of the ELBO over the runs.
    """
    elbo_values = []

    with open("elbo_values.txt", "w") as f:
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            # Reinitialize the model and optimizer for each run
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # Train the model
            train(model, optimizer, train_loader, epochs, device)

            # Evaluate ELBO
            total_elbo = 0.0
            num_batches = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    elbo = model.elbo(x.to(device))
                    total_elbo += elbo.item()
                    num_batches += 1
            avg_elbo = total_elbo / num_batches
            elbo_values.append(avg_elbo)

            # Write the ELBO value to the file
            f.write(f"Run {run + 1} ELBO: {avg_elbo:.4f}\n")

            print(f"Run {run + 1} ELBO: {avg_elbo:.4f}")

    # Compute mean and standard deviation of ELBO values
    mean_elbo = np.mean(elbo_values)
    std_elbo = np.std(elbo_values)

    # Write the mean and standard deviation to the file
    with open("elbo_values.txt", "a") as f:
        f.write(f"\nELBO over {num_runs} runs: Mean = {mean_elbo:.4f}, Std = {std_elbo:.4f}\n")

    print(f"ELBO over {num_runs} runs: Mean = {mean_elbo:.4f}, Std = {std_elbo:.4f}")
    
    return mean_elbo, std_elbo

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'train-multiple'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['flow', 'mog', 'gaussian'], help='type of prior to use (default: %(default)s)')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='M', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--num-components', type=int, default=3, metavar='K', help='number of MoG prior components (default: %(default)s)')
    parser.add_argument('--num-runs', type=int, default=10, help='number of runs for training and evaluating ELBO (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    


    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim

    # Build flow prior
    base = GaussianBase(M)
    transformations = []
    mask = torch.Tensor([i % 2 for i in range(M)])

    #mask[M//2:] = 1
    
    for i in range(10):
        mask = 1 - mask
        scale_net = nn.Sequential(nn.Linear(M, 256), nn.ReLU(), nn.Linear(256, M), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(M, 256), nn.ReLU(), nn.Linear(256, M))
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
    
    if args.prior == 'flow':
        flow = Flow(base, transformations)
        prior = FlowPrior(flow)

    elif args.prior == 'mog':
        prior = MoGPrior(M, args.num_components)

    else:
        prior = GaussianPrior(M)

    # Define encoder and decoder networks

    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)
    
    elif args.mode == 'eval':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        evalELBO(model, mnist_test_loader, args.device)
        
    elif args.mode == 'train-multiple':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train and evaluate multiple runs
        train_and_eval_multiple_runs(model, optimizer, mnist_train_loader, mnist_test_loader, args.epochs, args.device, num_runs=args.num_runs)
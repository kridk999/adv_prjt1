

import torch
import torch.nn as nn
import torch.distributions as td
import sys
import json
from pathlib import Path
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'week3'))
sys.path.insert(0, str(ROOT / 'week2'))

from ddpm import DDPM
from unet import Unet
from flow import Flow, GaussianBase, MaskedCouplingLayer
from fid import compute_fid

MODEL_DIR = Path(__file__).resolve().parent / 'Group_models'
CLASSIFIER_CKPT = str(ROOT / 'week3' / 'mnist_classifier.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 10000
M = 10   # latent dim
D = 784  # pixel dim

class FlowPrior(nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward(self):
        return self

    def sample(self, shape):
        return self.flow.sample(shape)

    def log_prob(self, z):
        return self.flow.log_prob(z)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net, D=784):
        super().__init__()
        self.decoder_net = decoder_net
        self.log_std = nn.Parameter(torch.zeros(D))

    def forward(self, z):
        mu = self.decoder_net(z)
        sigma = torch.exp(self.log_std)
        return td.Independent(td.Normal(loc=mu, scale=sigma), 1)


class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).mean


class FcNetwork4Layer(nn.Module):
    def __init__(self, input_dim, num_hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, input_dim),
        )

    def forward(self, x, t):
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


def build_vae(M=10):
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 2 * M),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784),
    )
    base = GaussianBase(M)
    transformations = []
    mask = torch.Tensor([i % 2 for i in range(M)])
    for _ in range(10):
        mask = 1 - mask
        scale_net = nn.Sequential(
            nn.Linear(M, 256), nn.ReLU(),
            nn.Linear(256, M), nn.Tanh(),
        )
        translation_net = nn.Sequential(
            nn.Linear(M, 256), nn.ReLU(),
            nn.Linear(256, M),
        )
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
    flow = Flow(base, transformations)
    prior = FlowPrior(flow)
    return VAE(prior, GaussianDecoder(decoder_net), GaussianEncoder(encoder_net))


def build_ddpm_unet(T=100):
    return DDPM(Unet(), T=T)


def build_latent_ddpm(M=10, T=100):
    return DDPM(FcNetwork4Layer(M, 512), T=T)

@torch.no_grad()
def sample_ddpm(model, n, dim, batch_size=500):
    model.eval()
    parts = []
    n_batches = (n + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, n, batch_size)):
        bs = min(batch_size, n - i)
        parts.append(model.sample((bs, dim)).cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def sample_vae(model, n, batch_size=2000):
    model.eval()
    parts = []
    for i in range(0, n, batch_size):
        bs = min(batch_size, n - i)
        parts.append(model.sample(bs).cpu())
    return torch.cat(parts, dim=0)


@torch.no_grad()
def sample_latent_ddpm(ddpm, vae, n, latent_dim, batch_size=2000):
    ddpm.eval(); vae.eval()
    parts = []
    n_batches = (n + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, n, batch_size)):
        bs = min(batch_size, n - i)
        z = ddpm.sample((bs, latent_dim))
        x = vae.decoder(z).mean
        parts.append(x.cpu())
    return torch.cat(parts, dim=0)

def to_fid_images(samples):
    mean_val = samples.mean().item()
    if mean_val > 0.2:
        imgs = samples.clamp(0, 1).view(-1, 1, 28, 28)
        imgs = (imgs - 0.5) * 2.0
        tag = "[0,1]"
    else:
        imgs = samples.clamp(-1, 1).view(-1, 1, 28, 28)
        tag = "[-1,1]"
    return imgs, tag


def load_real_images(n=N_SAMPLES):
    dataset = datasets.MNIST(root=str(ROOT / 'data'), train=False,
                             download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=n, shuffle=False)
    images, _ = next(iter(loader))
    images = images[:n]
    images = (images - 0.5) * 2.0
    return images


if __name__ == '__main__':
    results = {}

    real_images = load_real_images().to(DEVICE)

    ddpm = build_ddpm_unet(T=100).to(DEVICE)
    ddpm.load_state_dict(torch.load(MODEL_DIR / 'mnist_ddpm_model.pt', map_location=DEVICE))
    samples = sample_ddpm(ddpm, N_SAMPLES, D, batch_size=500)
    gen, tag = to_fid_images(samples)
    fid = compute_fid(real_images, gen.to(DEVICE), device=DEVICE, classifier_ckpt=CLASSIFIER_CKPT)
    results['DDPM (Unet, T=100)'] = round(float(fid), 4)
    del ddpm; torch.cuda.empty_cache()

    betas = {'b1': 1.0, 'b1e-6': 1e-6, 'b0.5': 0.5}

    for label, beta_val in betas.items():
        vae = build_vae(M).to(DEVICE)
        vae.load_state_dict(torch.load(MODEL_DIR / f'{label}vae_model.pt', map_location=DEVICE))

        vae_samples = sample_vae(vae, N_SAMPLES)
        gen_vae, tag = to_fid_images(vae_samples)
        fid_vae = compute_fid(real_images, gen_vae.to(DEVICE), device=DEVICE, classifier_ckpt=CLASSIFIER_CKPT)
        results[f'VAE (β={beta_val})'] = round(float(fid_vae), 4)

        ldm = build_latent_ddpm(M, T=100).to(DEVICE)
        ldm.load_state_dict(torch.load(MODEL_DIR / f'{label}latent_ddpm_model.pt', map_location=DEVICE))
        ldm_samples = sample_latent_ddpm(ldm, vae, N_SAMPLES, M, batch_size=2000)
        gen_ldm, tag = to_fid_images(ldm_samples)
        fid_ldm = compute_fid(real_images, gen_ldm.to(DEVICE), device=DEVICE, classifier_ckpt=CLASSIFIER_CKPT)
        results[f'Latent DDPM (β={beta_val})'] = round(float(fid_ldm), 4)

        del vae, ldm; torch.cuda.empty_cache()

    out_path = Path(__file__).resolve().parent / 'fid_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

import torch, torch.nn as nn, torch.distributions as td
import sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'week3'))
sys.path.insert(0, str(ROOT / 'week2'))
from flow import Flow, GaussianBase, MaskedCouplingLayer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = Path(__file__).resolve().parent / 'Group_models'
OUT_DIR = ROOT / 'assignment_plots'
OUT_DIR.mkdir(exist_ok=True)
M, N_SAMPLES = 10, 10000


class DDPM_RawT(nn.Module):
    def __init__(self, network, T=100):
        super().__init__()
        self.network = network; self.T = T
        self.beta = nn.Parameter(torch.linspace(1e-4, 2e-2, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

    @torch.no_grad()
    def sample(self, shape):
        x_t = torch.randn(shape).to(self.alpha.device)
        for t in range(self.T - 1, -1, -1):
            z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            t_b = torch.full((shape[0], 1), t, device=x_t.device, dtype=torch.float32)
            x_t = (1 / torch.sqrt(self.alpha[t])) * (
                x_t - ((1 - self.alpha[t]) / torch.sqrt(1 - self.alpha_cumprod[t]))
                * self.network(x_t, t_b)) + torch.sqrt(self.beta[t]) * z
        return x_t


class FlowPrior(nn.Module):
    def __init__(self, flow):
        super().__init__(); self.flow = flow
    def forward(self): return self
    def sample(self, shape): return self.flow.sample(shape)

class GaussianEncoder(nn.Module):
    def __init__(self, net):
        super().__init__(); self.encoder_net = net
    def forward(self, x):
        m, s = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(m, torch.exp(s)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, net, D=784):
        super().__init__()
        self.decoder_net = net; self.log_std = nn.Parameter(torch.zeros(D))
    def forward(self, z):
        return td.Independent(td.Normal(self.decoder_net(z), torch.exp(self.log_std)), 1)

class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super().__init__()
        self.prior = prior; self.decoder = decoder; self.encoder = encoder

class FcNetwork4Layer(nn.Module):
    def __init__(self, input_dim, num_hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, input_dim))
    def forward(self, x, t): return self.network(torch.cat([x, t], dim=1))


def build_vae():
    enc = nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 2 * M))
    dec = nn.Sequential(nn.Linear(M, 512), nn.ReLU(),
                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 784))
    base = GaussianBase(M)
    mask = torch.Tensor([i % 2 for i in range(M)])
    ts = []
    for _ in range(10):
        mask = 1 - mask
        ts.append(MaskedCouplingLayer(
            nn.Sequential(nn.Linear(M, 256), nn.ReLU(), nn.Linear(256, M), nn.Tanh()),
            nn.Sequential(nn.Linear(M, 256), nn.ReLU(), nn.Linear(256, M)), mask))
    return VAE(FlowPrior(Flow(base, ts)), GaussianDecoder(dec), GaussianEncoder(enc))


@torch.no_grad()
def get_aggregate_posterior(vae, data_loader, n_max=N_SAMPLES):
    vae.eval(); codes = []
    for x, _ in data_loader:
        codes.append(vae.encoder(x.to(DEVICE)).mean.cpu())
        if sum(c.shape[0] for c in codes) >= n_max: break
    return torch.cat(codes)[:n_max]

@torch.no_grad()
def get_prior_samples(vae, n=N_SAMPLES):
    vae.eval(); return vae.prior().sample(torch.Size([n])).cpu()

@torch.no_grad()
def get_ddpm_samples(ddpm, n=N_SAMPLES):
    ddpm.eval(); return ddpm.sample((n, M)).cpu()


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=str(ROOT / 'data'), train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - 0.5) * 2.0),
                transforms.Lambda(lambda x: x.squeeze())])),
        batch_size=500, shuffle=False)

    vae = build_vae().to(DEVICE)
    vae.load_state_dict(torch.load(MODEL_DIR / 'b1e-6vae_model.pt', map_location=DEVICE))

    ldm = DDPM_RawT(FcNetwork4Layer(M, 512), T=100).to(DEVICE)
    ldm.load_state_dict(torch.load(MODEL_DIR / 'b1e-6latent_ddpm_model.pt', map_location=DEVICE))

    agg = get_aggregate_posterior(vae, train_loader)
    prior = get_prior_samples(vae)
    ddpm_z = get_ddpm_samples(ldm)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5),
                             gridspec_kw={'height_ratios': [1.1, 1]})
    C_AGG, C_PRI, C_DDPM = '#2176AE', '#E67E22', '#27AE60'
    N_PTS = 3000

    all_data = torch.cat([agg, prior, ddpm_z])
    lo = all_data[:, [0, 1]].min().item() * 1.25
    hi = all_data[:, [0, 1]].max().item() * 1.25

    for ax, (data, title, c) in zip(axes[0], [
        (agg, 'Aggregate posterior $q(z)$', C_AGG),
        (prior, 'Flow prior $p(z)$', C_PRI),
        (ddpm_z, 'Latent DDPM', C_DDPM),
    ]):
        ax.scatter(data[:N_PTS, 0].numpy(), data[:N_PTS, 1].numpy(),
                   s=2, alpha=0.3, c=c, rasterized=True)
        ax.set(xlabel='$z_{0}$', ylabel='$z_{1}$', xlim=(lo, hi), ylim=(lo, hi), aspect='equal')
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.15)

    lo_h = min(agg.min().item(), prior.min().item(), ddpm_z.min().item())
    hi_h = max(agg.max().item(), prior.max().item(), ddpm_z.max().item())
    bins = np.linspace(lo_h * 1.1, hi_h * 1.1, 60)

    for ax, d in zip(axes[1], [0, 4, 8]):
        ax.hist(agg[:, d].numpy(), bins=bins, density=True,
                alpha=0.55, color=C_AGG, label='Agg. posterior')
        ax.hist(prior[:, d].numpy(), bins=bins, density=True,
                alpha=0.45, color=C_PRI, label='Flow prior')
        ax.hist(ddpm_z[:, d].numpy(), bins=bins, density=True,
                alpha=0.40, color=C_DDPM, label='Latent DDPM')
        ax.set_xlabel(f'$z_{{{d}}}$')
        ax.set_ylabel('Density')
        ax.set_title(f'Marginal — dim {d}', fontsize=11)
        ax.tick_params(labelsize=8)

    axes[1, 0].legend(fontsize=8, loc='upper right')

    fig.suptitle(r'Latent space comparison ($\beta = 10^{-6}$)', fontsize=14, y=0.98)
    plt.tight_layout()

    fig.savefig(OUT_DIR / 'latent_distributions.png', dpi=200, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'latent_distributions.pdf', bbox_inches='tight')
    plt.close(fig)

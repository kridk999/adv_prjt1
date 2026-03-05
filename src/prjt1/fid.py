import argparse

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
from beta_vae_standard import VAE, GaussianEncoder, GaussianDecoder, GaussianPrior, FlowPrior
from MoGPrior import MoGPrior
from flow import Flow, GaussianBase, MaskedCouplingLayer
import unet
from ddpm import DDPM, FcNetwork
from torchvision.utils import save_image


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = torch.nn.Sequential(

            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),


            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout(0.25),

            torch.nn.Flatten(),
            torch.nn.Linear(9216, 128),
            torch.nn.Dropout(0.5),
        )

        self.classification_layer = torch.nn.Sequential(
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        y = self.layers(x)
        y = self.classification_layer(y)
        return y
    

def frechet_distance(x_a, x_b):
    mu_a = np.mean(x_a, axis=0)
    sigma_a = np.cov(x_a.T)
    mu_b = np.mean(x_b, axis=0)
    sigma_b = np.cov(x_b.T)

    diff = mu_a - mu_b
    covmean = scipy.linalg.sqrtm(sigma_a @ sigma_b)
    return np.sum(diff**2) + np.trace(sigma_a + sigma_b - 2.0 * covmean)


def compute_fid(
    x_real: torch.Tensor,
    x_gen: torch.Tensor,
    device: str = "cpu",
    classifier_ckpt: str = "mnist_classifier.pth",
) -> float:
    """Compute the Fréchet Inception Distance (FID) between two sets of images.
    
    Args:
        x_real (torch.Tensor): A batch of real images, shape (N, 1, 28, 28) and with values in [-1, 1].
        x_gen (torch.Tensor): A batch of generated images, shape (N, 1, 28, 28) and with values in [-1, 1].
        device (str): The device to run the classifier on ("cpu" or "cuda").
        classifier_ckpt (str): Path to the pre-trained classifier checkpoint

    Returns:
        float: The computed FID score between the two sets of images.
    """

    
    # ---- load classifier ----
    clf = Classifier().to(device)
    clf.load_state_dict(torch.load(classifier_ckpt, map_location=device))
    clf.eval()

    # ---- calculate latent features with classifier ----
    with torch.no_grad():
        real_latent = clf.layers(x_real)
        gen_latent = clf.layers(x_gen)
    real_latent = real_latent.cpu().numpy()
    gen_latent = gen_latent.cpu().numpy()

    return frechet_distance(real_latent, gen_latent)


if __name__ == '__main__':
    from torchvision import datasets, transforms
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='mnist', choices=['tg', 'cb', 'mnist', 'latent-space'], help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=10, metavar='N', help='dimension of the latent space (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='gaussian', choices=['mog', 'gaussian', 'flow'], help='prior to use for the VAE (default: %(default)s)')
    parser.add_argument('--bvae-model', type=str, default='b0.5vae_model.pt', help='file to load the VAE model from (default: %(default)s)')
    parser.add_argument('--num-transformations', type=int, default=40, help='number of transformations for flow prior (default: %(default)s)')    
        
    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)
        
        
            
    mnist_dataset = datasets.MNIST('data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms . Lambda ( lambda x : (x -0.5) *2.0),  
            transforms.Lambda(lambda x: x.squeeze())  # (1,28,28) → (28,28)

        ]))
    
    train_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=args.batch_size, shuffle=False)
        
    if args.data == 'latent-space':
        D = next(iter(train_loader))[0].shape[1]
    else:
        D = next(iter(train_loader)).shape[1]
        
        # Set the number of steps in the diffusion process
    T = 1000
    
        # Define the network
    if args.data == 'latent-space':
        num_hidden = 1024
        network = FcNetwork(args.latent_dim, num_hidden)
    else:
        network = unet.Unet()

    # Define model
    model = DDPM(network, T=T).to(args.device)

    if args.data == 'latent-space':
        M = args.latent_dim

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
            #nn.Unflatten(-1, (28, 28))
        )
        encoder = GaussianEncoder(encoder_net)
        decoder = GaussianDecoder(decoder_net)

    
    
    if args.prior == 'mog':
            prior = MoGPrior(M, K=7)
    elif args.prior == 'gaussian':
        prior = GaussianPrior(M)
    elif args.prior == 'flow':
        base = GaussianBase(M)
        transformations = []
        mask = torch.Tensor([i % 2 for i in range(M)])

        #mask[M//2:] = 1
        
        for i in range(args.num_transformations):
            mask = 1 - mask
            scale_net = nn.Sequential(nn.Linear(M, 256), nn.ReLU(), nn.Linear(256, M), nn.Tanh())
            translation_net = nn.Sequential(nn.Linear(M, 256), nn.ReLU(), nn.Linear(256, M))
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

        flow = Flow(base, transformations)
        prior = FlowPrior(flow)

    
    bvae_model = VAE(prior, decoder, encoder).to(args.device)
    bvae_model.load_state_dict(torch.load(args.bvae_model, map_location=torch.device(args.device)))
    bvae_model.eval()
    
    #thresshold = 0.5
    #binarized_data = (thresshold < toy().sample((n_data,))).float()
    #train_loader = torch.utils.data.DataLoader(binarized_data, batch_size=args.batch_size, shuffle=True)
    
    print("Encoding dataset to latents...")
    latent_data = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(args.device)
            # Get the distribution and sample z
            q = bvae_model.encoder(x)
            z = q.sample()
            latent_data.append(z.cpu())
    
    latent_dataset = torch.cat(latent_data, dim=0)
    
    train_loader_latents = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(latent_dataset), 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    if args.mode == 'sample':
        import numpy as np

        # Load the model
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            
            if args.data == 'mnist':
                # Reshape to 28x28 images and clamp to [0, 1]
                #display only 64 samples in an 8x8 grid
                samples = (model.sample((args.batch_size,D))).cpu() 
                samples = samples.view(-1, 1, 28, 28).clamp(0, 1)

            elif args.data == 'latent-space':
                samples = (model.sample((args.batch_size,M))).cpu() 
                bvae_model.eval()
                with torch.no_grad():
                    decoder_dist = bvae_model.decoder(samples.to(args.device))
                    samples = decoder_dist.mean.cpu()  # ← mean, no decoder noise
                samples = samples.view(-1, 1, 28, 28)
                samples = (samples / 2 + 0.5).clamp(0, 1)
                save_image(samples, args.samples, nrow=8)
    
    print("Decoding MNIST test data using VAE decoder...")
    with torch.no_grad():
        x_test = next(iter(test_loader))[0].to(args.device)
        q = bvae_model.encoder(x_test)
        z = q.mean  # ← deterministic, fair reconstruction comparison
        vae_decoded = bvae_model.decoder(z).mean.cpu()  # ← mean, no decoder noise
        vae_decoded = vae_decoded.view(-1, 1, 28, 28)
        vae_decoded = (vae_decoded / 2 + 0.5).clamp(0, 1)
        save_image(vae_decoded, "vae_decoded_samples.png", nrow=8)

    
    # Compute FID for VAE decoded samples
    vae_fid_score = compute_fid(
        (x_test.unsqueeze(1) / 2 + 0.5).cpu(),  # real images, rescaled
        vae_decoded.cpu(),
        device='cpu'
    )
    with open(f"fid_score_vae_{args.data}_{args.prior}.txt", "w") as f:
        f.write(f"VAE Decoder FID score: {vae_fid_score:.4f}\n")
        f.write(f"Model: {args.bvae_model}\n")
        f.write(f"Prior: {args.prior}\n")
        f.write(f"Latent dim: {args.latent_dim}\n")
    print(f"VAE Decoder FID score: {vae_fid_score:.4f}")


    fid_score = compute_fid(
        (next(iter(test_loader))[0].unsqueeze(1) / 2 + 0.5).to('cpu'),  # rescale [-1,1] → [0,1]
        samples.cpu(),
        device='cpu'
    )
    
    with open(f"fid_score_ddpm_{args.data}_{args.prior}.txt", "w") as f:
        f.write(f"FID score: {fid_score:.4f}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Prior: {args.prior}\n")
        f.write(f"Latent dim: {args.latent_dim}\n")
    
    print(f"FID score: {fid_score:.4f}")
    
    
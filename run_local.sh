#!/bin/bash
source .venv/bin/activate

# Train VAE (higher beta for smoother latent space)
uv run src/prjt1/beta_vae_standard.py train \
    --prior flow \
    --latent-dim 16 \
    --num-transformations 20 \
    --beta 0.5 \
    --model b0.5flow16_vae_model.pt \
    --epochs 2 \
    --device mps

# Train latent DDPM (lower lr, more epochs)
uv run src/prjt1/ddpm.py train \
    --data latent-space \
    --device mps \
    --epochs 10 \
    --lr 1e-4 \
    --latent-dim 16 \
    --prior flow \
    --bvae-model b0.5flow16_vae_model.pt \
    --num-transformations 20 \
    --model ddpm_flow16_model.pt

# Train normal DDPM
uv run src/prjt1/ddpm.py train \
    --data mnist \
    --device mps \
    --epochs 10 \
    --model ddpm_mnist_model.pt

# Sample and compute FID
uv run src/prjt1/fid.py sample \
    --data latent-space \
    --prior flow \
    --latent-dim 16 \
    --bvae-model b0.5flow16_vae_model.pt \
    --device mps \
    --num-transformations 20 \
    --model ddpm_flow16_model.pt

# Sample and compute FID
uv run src/prjt1/fid.py sample \
    --data mnist \
    --device mps \
    --model ddpm_mnist_model.pt \
    --sample mnist_ddpm.png
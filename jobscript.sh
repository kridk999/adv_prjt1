#!/bin/sh




### General LSF options

# Jobname
# BSUB -J_VAE_TEST

### –- specify queue --
#BSUB -q gpuv100

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 10:00

# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- send notification at start --
##BSUB -B

### -- send notification at completion--
##BSUB -N

### -- end of LSF options --

# Load environment variables
source ./.env


# Activate venv
module load python3/3.12.4
module load cuda/12.4
source .venv/bin/activate

#BSUB -o Output_%J.out
#BSUB -e Output_%J.err


# run training
#python3 src/prjt1/vae_bernoulli.py train-multiple --epochs 2 --prior gaussian --num-runs 2 --device cuda
# Train VAE (higher beta for smoother latent space)
time python3 src/prjt1/beta_vae_standard.py train \
    --prior flow \
    --latent-dim 16 \
    --num-transformations 20 \
    --beta 0.5 \
    --model b0.5flow16_vae_model.pt \
    --epochs 50 \
    --device mps

# Train latent DDPM (lower lr, more epochs)
time python3 src/prjt1/ddpm.py train \
    --data latent-space \
    --device mps \
    --epochs 100 \
    --lr 1e-4 \
    --latent-dim 16 \
    --prior flow \
    --bvae-model b0.5flow16_vae_model.pt \
    --num-transformations 20 \
    --model ddpm_flow16_model.pt

# Sample and compute FID
time python3 src/prjt1/fid.py sample \
    --data latent-space \
    --prior flow \
    --latent-dim 16 \
    --bvae-model b0.5flow16_vae_model.pt \
    --device mps \
    --num-transformations 20 \
    --model ddpm_flow16_model.pt \
    --sample ddpm_samples.png
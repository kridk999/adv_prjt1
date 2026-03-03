#!/bin/sh




### General LSF options

# Jobname
# BSUB -J_VAE_TEST

### –- specify queue --
#BSUB -q gpuv100

### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00

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
python3 src/prjt1/beta_vae_standard.py train --model bvae_model.pt --epochs 50 --prior flow --device cuda --beta 0.5

python3 src/prjt1/ddpm.py train --data latent-space --epochs 100 --prior flow --device cuda --model latent_ddpm_model.pt

python3 src/prjt1/ddpm.py sample --data latent-space --model latent_ddpm_model.pt
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
python3 src/prjt1/vae_bernoulli.py train-multiple --epochs 2 --prior gaussian --num-runs 2 --device cuda
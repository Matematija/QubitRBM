#!/bin/bash

#SBATCH -N1 --exclusive
#SBATCH --job-name=QubitRBM1.job
#SBATCH --output=.out/QubitRBM_1.out
#SBATCH --error=.out/QubitRBM_1.err
#SBATCH --mem=16000

module purge
module load slurm gcc python3

cd ..

virtualenv venv
source venv/bin/activate

pip3 install numpy, scipy
pip3 install joblib

pip3 freeze

cd cluster

python3 twenty_qubits.py

deactivate

rm -rf venv
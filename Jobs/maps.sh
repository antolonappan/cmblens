#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=5
#SBATCH --ntasks=50
#SBATCH -J CMBLensed
#SBATCH -o maps.out
#SBATCH -e maps.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

module load python

conda activate

conda activate PC2

cd /global/u2/l/lonappan/workspace/CMBlens/cmblens


mpirun -np $SLURM_NTASKS python cmblens.py
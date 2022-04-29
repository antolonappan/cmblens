#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=5
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=2
#SBATCH -J CMBLensed
#SBATCH -o out/maps.out
#SBATCH -e out/maps.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/cmblens/cmblens


mpirun -np $SLURM_NTASKS python cmblens.py
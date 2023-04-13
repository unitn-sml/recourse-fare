#!/usr/bin/env bash
#PBS -l select=2:ncpus=35:mem=5GB:mpiprocs=25
#PBS -l walltime=5:59:59
#PBS -q short_cpuQ
#PBS -M giovanni.detoni@unitn.it
#PBS -V
#PBS -m be

# https://github.com/open-mpi/ompi/issues/7701
export HWLOC_COMPONENTS=-gl

export PATH=$HOME/miniconda3/bin:$PATH
source ~/.bashrc

cd recourse-fare

git checkout feature/add-w-fare

module load openmpi-3.0.0

conda activate rl_mcts

export PYTHONPATH=./

mpirun python3 recourse_fare/example/wfare/experiment.py --questions "${COUNTER}"

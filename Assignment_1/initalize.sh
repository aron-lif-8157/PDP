#!/bin/bash
ml gcc openmpi
mpicc -o sum sum.c
sbatch ./submit.sh
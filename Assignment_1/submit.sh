#!/bin/bash
#SBATCH -A uppmax2025-2-247
#SBATCH -J arli8157-Ass_1
#SBATCH -n 8
#SBATCH --time=00:05:00
#SBATCH --error=Ass_1.%J.err
#SBATCH --output=Ass_1.%J.out


for scaling in "strong" "weak"; do
    echo "------- $scaling Scaling -----"
    for step in {22..26}; do
        for procs in {1..8}; do
            echo "$procs process - 2^$step $scaling"
            mpirun -np $procs ./sum $step $scaling
        done
    done
done

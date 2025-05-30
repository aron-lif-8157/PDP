#!/bin/bash
#SBATCH --job-name=malaria_scaling
#SBATCH --nodes=2
#SBATCH --account=uppmax2025-2-247
#SBATCH --partition=node

# Load MPI module (change this to match your cluster)
module load intelmpi/20.4   # Replace with actual module name

# Make the script executable
chmod +x run_scaling.sh

# Run the scaling test
bash run_scaling.sh
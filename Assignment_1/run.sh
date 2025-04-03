
echo "cleaning build"
make clean

echo "building"
make
echo "OK"

echo "running"
echo "running with 1 process sum 19"

# Define the process counts and the num_steps values you want to test.
process_counts=(1 2 3 4 5 6 7 8)
sum_exponents=(19 20 21 22)

# Loop over each process count
for p in "${process_counts[@]}"; do
  # Loop over each exponent value (num_steps)
  for exp in "${sum_exponents[@]}"; do
    echo "Running with $p processes and num_steps = $exp"
    # Run the test a few times to gather multiple measurements
    for run in {1..4}; do
      echo "Run $run:"
      mpirun -np "$p" ./sum "$exp"
    done
    echo ""
  done
done
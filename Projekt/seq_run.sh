set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_runs>"
    exit 1
fi

RUNS="$1"

BASE="./output/sequentialruns_${RUNS}"
SUFFIX=1
OUTFILE="${BASE}_${SUFFIX}.csv"

# bump SUFFIX until we find a free name
while [ -e "$OUTFILE" ]; do
  SUFFIX=$((SUFFIX+1))
  OUTFILE="${BASE}_${SUFFIX}.csv"
done

echo "Using output file $OUTFILE"

echo "cleaning build"
make clean

echo "building project"
make

echo "Running $RUNS sequential simulations → $OUTFILE"
./build/main "$OUTFILE" "$RUNS"
echo "Done."

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# benchmark.sh – sweep cores × pivot × input size for quicksort (Assignment 3)
# ---------------------------------------------------------------------------
# * Assumes the executable ./quicksort is already built (see Makefile).
# * Works on any cluster where 'mpirun' launches Open MPI jobs.
# * Creates/updates results.csv:  cores,pivot,input,parallel_time,serial_time,speedup
# ---------------------------------------------------------------------------

# ---- parameters you might want to tweak -----------------------------------
CORES_LIST=(2 4 8)                 # outer loop
PIVOT_LIST=(1 2 3)                    # middle loop
INPUT_FILES=(
  ../inputs/backwards/input_backwards125000000.txt
  ../inputs/backwards/input_backwards250000000.txt
  ../inputs/backwards/input_backwards500000000.txt
  ../inputs/backwards/input_backwards1000000000.txt
  ../inputs/backwards/input_backwards2000000000.txt
)
OUTFILE_PREFIX="/dev/null"            # we do not need the sorted output
CSV=results.csv                       # aggregated results
# ---------------------------------------------------------------------------

# header (create file on first run)
if [[ ! -f $CSV ]]; then
    echo "cores,pivot,input,parallel_time,serial_time,speedup" > "$CSV"
fi

for PIVOT in "${PIVOT_LIST[@]}"; do
    for FILE in "${INPUT_FILES[@]}"; do
      SERIAL_TIME=$( mpirun -np 1 ./quicksort "$FILE" "$OUTFILE_PREFIX" "$PIVOT" | tail -n 1 )

      for CORES in "${CORES_LIST[@]}"; do


      # ------------- serial baseline ---------------------------------------

      # ------------- parallel run ------------------------------------------
      PAR_TIME=$( mpirun -np "$CORES" ./quicksort "$FILE" "$OUTFILE_PREFIX" "$PIVOT" | tail -n 1 )

      # ------------- speed-up ----------------------------------------------
      SPEEDUP=$( echo "scale=3; $SERIAL_TIME / $PAR_TIME" | bc )

      # ------------- report & log ------------------------------------------
      printf "%2d-cores | pivot %d | %-28s | Tpar=%8s s | Tser=%8s s | S=%5s×\n" \
             "$CORES" "$PIVOT" "$FILE" "$PAR_TIME" "$SERIAL_TIME" "$SPEEDUP"

      echo "${CORES},${PIVOT},${FILE},${PAR_TIME},${SERIAL_TIME},${SPEEDUP}" >> "$CSV"
    done
  done
done

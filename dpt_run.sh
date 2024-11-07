#! /bin/bash

for i in {1..2}; do
    echo "Iteration ${i}..."
    ./run.sh $i
done

python3 ./solvers/dpt/train_dpt.py
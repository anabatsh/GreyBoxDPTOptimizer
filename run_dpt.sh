#! /bin/bash

for i in {1..100}; do
    echo "Problem ${i}..."
    ./run.sh $i
done

# python3 ./solvers/dpt/train_dpt.py
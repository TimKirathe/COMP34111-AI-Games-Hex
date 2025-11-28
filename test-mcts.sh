#!/bin/bash

p1="$1"
p2="$2"
num_runs="$3"

for run in $(seq "$(($num_runs / 2))"); do
    echo "Run: $run; p1: $p1 (Alice); p2: $p2 (Bob)"
    docker run --cpus=8 --memory=8G -v "$(pwd)":/home/hex --rm hex python3 Hex.py -p1 "$p1" -p2 "$p2" >tmp_run.log 2>&1
    echo "$(tail -n 3 tmp_run.log)"
    rm tmp_run.log
done

for run in $(seq "$(($num_runs / 2))" "$num_runs"); do
    echo "Run: $run; p1: $p2 (Alice); p2: $p1 (Bob)"
    docker run --cpus=8 --memory=8G -v "$(pwd)":/home/hex --rm hex python3 Hex.py -p1 "$p2" -p2 "$p1" >tmp_run.log 2>&1
    echo "$(tail -n 3 tmp_run.log)"
    rm tmp_run.log
done

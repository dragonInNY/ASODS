#!/bin/bash

#SBATCH --account=free

times=()

for i in {28124256..28124268}; do
    output_file="slurm-${i}.out"  
    output=$(grep -oP 'Duration:\s*\K[\d.]*' "$output_file")  
    times+=("$output")
done

echo "${times[*]}" | tr ' ' ','
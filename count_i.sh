#!/bin/bash

#SBATCH --account=free


slurm_out_file="/rigel/home/hy2726/slurm-28124144.out"

declare -A counts

# Read the file line by line
while IFS= read -r line; do

    if [[ $line == i:* ]]; then

        number=${line#i: }

        ((counts[$number]++))
    fi
done < "$slurm_out_file"

for number in "${!counts[@]}"; do
    echo "$number: ${counts[$number]}"
done

#!/bin/sh
#SBATCH --account=free    

job_name="TRMR"

PARAMS=()
for ((i=96; i>=48; i = i-4))
do
    PARAMS+=($i)
done

for param in "${PARAMS[@]}"
do
    CMD="source activate ASODS && mpirun -np $param python /rigel/home/hy2726/codes/tests/${job_name}_test.py"

    sbatch --job-name="$job_name " <<EOT
#!/bin/sh
 
#SBATCH --account=free  
#SBATCH -N $(((param+23)/24))
#SBATCH --exclusive
#SBATCH --time=1:00:00 

$CMD
EOT

done
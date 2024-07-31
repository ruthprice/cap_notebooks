#!/bin/sh
module load jaspy

suites_f_in='suites.csv'
stash_f_in='stashcodes.csv'
n=0
while IFS=',' read -r suite config
do
  while read stash
    do
      echo ${suite}, ${config}, ${stash}
      rm -f logs/log_${suite}_${stash}*
      sbatch -o logs/log_${suite}_${stash}.out -e logs/log_${suite}_${stash}.err --partition=short-serial-4hr --account=short4hr --time=00:30:00 --job-name=${suite}_${stash} capricorn-collocate-xr.py --suite=${suite} --config="${config}" --stashcode=${stash}
      n=$((n+1))
  done < ${stash_f_in}
done < ${suites_f_in}

echo "Submission loop finished! Submitted $n times"

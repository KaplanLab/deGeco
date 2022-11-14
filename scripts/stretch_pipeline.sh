#!/bin/bash
set -e
mcool_filename=${mcool_filename:-~/storage/Rao_GM12878_zoomified.mcool}
chr=${chr:-all_no_ym}
mkdir -p logs
mkdir -p output

if [ -z "$1" -o "$1" = --help ]; then
    echo Usage: $0 STATES
    echo
    echo Will fit all chromosomes except Y and M from mcool $mcool_filename using this procedure:
    echo - Fit 20 times at binsize=500kb '(in parallel)'
    echo - Take the best 5
    echo - Refine the taken fits '(in parallel)' at resolutions: 100kb, 50kb
    echo - Take the best 50kb fit
    echo 
    echo Restarting an interrupted run will skip previously completed steps.
    echo Note that running many fits in parallel can have large memory and CPU requirements.
    exit 1
fi

st=$1

echo Running initial low-res fit
for i in {1..20}; do
    basename=${chr}_${st}st_500000_run${i}
    python gc_model_main.py -m $mcool_filename -ch ${chr} -kb 500000 -o output/${basename}.npz -n $st --seed $i --iterations 1 --no-overwrite > logs/${basename}.log 2>&1 &
done
wait

python get_best.py output/${chr}_${st}st_500000_run*.npz -n 5 --target "output/${chr}_${st}st_500000_top{n}.npz"

for i in {1..5}; do
    (
    echo Fit $i: Refining
    prev_basename=${chr}_${st}st_500000_top${i}
    basename=${chr}_${st}st_100000_run${i}
    python gc_model_main.py -m $mcool_filename -ch ${chr} -kb 100000 -o output/${basename}.npz -n $st --seed $i --iterations 1 --init output/${prev_basename}.npz --init-stretch-by 5 --no-overwrite > logs/${basename}.log 2>&1

    prev_basename=$basename
    basename=${chr}_${st}st_50000_run${i}
    # Zero sample is the number of non-zero entries at 50kb for this mcool
    python gc_model_main.py -m $mcool_filename -ch ${chr} -kb 50000 -o output/${basename}.npz -n $st --seed $i --iterations 1 --sparse --zero-sample 720607320 --init output/${prev_basename}.npz --init-stretch-by 2 --no-overwrite > logs/${basename}.log 2>&1
    echo Fit $i: Done
) &
done

wait

python get_best.py output/${chr}_${st}st_50000_run*.npz --target "output/${chr}_${st}st_50000_best.npz"

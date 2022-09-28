#!/bin/bash
set -eE
#mcool_filename=~/storage/Rao_GM12878_zoomified.mcool
mcool_filename=data_files/hic/Rao2014-GM12878-MboI-allreps-filtered.mcool
chr=chr19
st=2
allgenome_reads=(100000 250000 500000 750000 1000000)
chr_reads_ratio=0.02 # Manually calculated by dividing unbalanced lower triangle of chr by lower triangle of full matrix
resolutions=(1000000 40000,500000 250000,100000 50000) # Comma means multiple resolutions that are stretched from the same source. Last one is used for further stretching.
iterations=10
out_dir=output
log_dir=logs

if [ "$1" = --help ]; then
    echo Usage: $0
    echo
    echo Will simulate single-cell experiments using $chr data from mcool $mcool_filename using this procedure:
    echo - Create ground-truth fits at resolutions: ${resolutions[@]}
    echo - Sample ground-truth fits to simulate scHi-C experiments with these read counts for all genome: ${allgenome_reads[@]}
    echo - For each resolution, fit $iterations times, initializing each run with the solution from the previous resolution '(in parallel)'
    echo - For each resolution, choose the best fit using the achieved log likelihood score.
    echo
    echo Output files will be put in an $out_dir/ directory. Log files will be written to a $log_dir/ directory.
    echo Restarting an interrupted run will skip previously completed steps.
    echo Note that running many fits in parallel can have large memory and CPU requirements.
    exit 1
fi

mkdir -p $log_dir/{origin,sampled,fit}
mkdir -p $out_dir/{origin,sampled,fit}

echo Creating ground-truth fits for resolutions: ${resolutions[@]}
for res_batch in ${resolutions[@]}; do
    for r in `echo $res_batch | tr , ' '`; do
        res_basename=${chr}_${st}st_${r}
        for i in `seq $iterations`; do
            (
            basename=${res_basename}_run${i}
            python gc_model_main.py -m $mcool_filename -ch $chr -kb $r -n $st --seed $i -o $out_dir/origin/${basename}.npz -s symmetric --iterations 1 --no-overwrite > logs/origin/${basename}.log 2>&1 
            ) &
        done
        wait
        python get_best.py $out_dir/origin/${res_basename}_run*.npz --target "$out_dir/origin/${res_basename}_best.npz"
    done
done

echo Sampling ground-truth fits
for res_batch in ${resolutions[@]}; do
    for res in `echo $res_batch | tr , ' '`; do
        origin=$out_dir/origin/${chr}_${st}st_${res}_best.npz
        for reads in ${allgenome_reads[@]}; do
            (
            basename=${chr}_${st}st_res${res}_reads${reads}
            chr_reads=`bc <<<"$chr_reads_ratio * $reads /1"`
            python resample.py --reads $chr_reads -f $origin --seed 1 -o ${out_dir}/sampled/${basename}.npy --balance no > logs/sampled/${basename}.log 2>&1
            ) &
        done
        wait
    done
done

echo Fitting sampled data
for reads in ${allgenome_reads[@]}; do
    for i in `seq $iterations`; do
        (
            previous_basename=""
            previous_res=""
            for res_batch in ${resolutions[@]}; do
                for res in `echo $res_batch | tr , ' '`; do
                    res_basename=${chr}_${st}st_res${res}_reads${reads}
                    fit_basename=${res_basename}_run${i}
                    if [ -z "$previous_basename" ]; then
                        echo reads=$reads: iteration $i: Running initial low-res fit $res
                        stretch_args=""
                    else
                        echo reads=$reads: iteration $i: Refining resolution $previous_res to $res
                        stretch_args="--init $out_dir/fit/${previous_basename}.npz --init-stretch-by $((previous_res/res))"
                    fi
                    python gc_model_main.py -m $out_dir/sampled/${res_basename}.npy -ch ${chr} -kb ${res} -o $out_dir/fit/${fit_basename}.npz -n $st --seed $i --iterations 1 --no-overwrite $stretch_args > $log_dir/fit/${fit_basename}.log 2>&1
                done
                previous_basename=$fit_basename
                previous_res=$res
            done
        ) &
    done
    wait
    for res_batch in ${resolutions[@]}; do
        for res in `echo $res_batch | tr , ' '`; do
            fit_basename=${chr}_${st}st_res${res}_reads${reads}
            python get_best.py $out_dir/fit/${fit_basename}_run*.npz --target "$out_dir/fit/${fit_basename}_best.npz"
        done
    done
done
wait

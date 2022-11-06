#!/bin/bash
set -eE
mcool_filename=~/storage/Rao_GM12878_zoomified.mcool
chr=chr19
sampling_rates=(1.0 0.75 0.5 0.25 0.1 0.05 0.01 0.005 0.001 0.0005)
resolutions=(10000 20000 100000 500000)
resolutions_asc=(`echo ${resolutions[@]} | tr ' ' '\n' | sort -n `)
resolutions_desc=(`echo ${resolutions[@]} | tr ' ' '\n' | sort -nr`)
iterations=10

if [ "$1" = --help ]; then
    echo Usage: $0 '[OUTPUT_DIR [STATES]]'
    echo
    echo Will downsample and fit $chr mcool $mcool_filename using this procedure:
    echo - Downsample at rates: ${sampling_rates[@]}
    echo - Bin to resolutions: ${resolutions_desc[@]}
    echo - For each resolution, fit $iterations times, initializing each run with the solution from the previous resolution '(in parallel)'
    echo - For each resolution, choose the best fit using the achieved log likelihood score.
    echo
    echo Output files will be put in an output/ directory inside OUTPUT_DIR, or current dir if this is not given. OUTPUT_DIR will be created if it does not exist.
    echo Fits will be done using STATES states, of 2 if unspecified.
	echo Log files will be written to a logs/ directory inside OUTPUT_DIR.
    echo Restarting an interrupted run will skip previously completed steps.
    echo Note that running many fits in parallel can have large memory and CPU requirements.
    exit 1
fi

output_dir=$1
[ -z "$output_dir" ] && output_dir=.
st=$2
[ -z "$st" ] && st=2
mkdir -p $output_dir/logs
mkdir -p $output_dir/output

base_res=${resolutions_asc[0]}
res_joined=`echo ${resolutions_asc[@]} | tr ' ' ,`
echo Downsampling at rates ${sampling_rates} and resolutions ${resolutions[@]}
for s in ${sampling_rates[@]}; do
    (
    baseres_basename=${chr}_downsampled_${s}_${base_res}_unbalanced
    binned_basename=${chr}_downsampled_${s}
    binned_mcool=$output_dir/output/${binned_basename}.mcool
    if [ -e "$binned_mcool" ]; then
        echo fraction=$s: Found existing binned mcool, skipping.
        exit 0
    fi
    echo fraction=$s: Downsampling at base resolution $base_res
    python -u downsample.py -m "$mcool_filename" -c "$chr" -r ${base_res} -o $output_dir/output/${baseres_basename}.cool -s $s --seed 1 --balance no > $output_dir/logs/${baseres_basename}.log 2>&1
    echo fraction=$s: Binning and balancing
    cooler zoomify -o $binned_mcool -p 5 -c 5000000 --balance --balance-args '--max-iters 10000' -r "$res_joined"  "$output_dir/output/${baseres_basename}.cool" > $output_dir/logs/${binned_basename}.log 2>&1
    cooler balance $binned_mcool::/resolutions/$base_res --max-iters 10000 --force >> $output_dir/logs/${binned_basename}.log 2>&1
    echo fraction=$s: Downsampled mcool is in $binned_mcool
    )&
done
wait

echo Fitting all resolutions and sampling rates
for s in ${sampling_rates[@]}; do
    binned_basename=${chr}_downsampled_${s}
    binned_mcool=$output_dir/output/${binned_basename}.mcool
    for i in `seq $iterations`; do
        (
            previous_basename=""
            previous_r=""
            for r in ${resolutions_desc[@]}; do
                fit_basename=${chr}_${st}st_s${s}_${r}_run${i}
                    if [ -z "$previous_basename" ]; then
                        echo fraction=$s: iteration $i: Running initial low-res fit $r
                        stretch_args=""
                    else
                        echo fraction=$s: iteration $i: Refining at resolution $r
                        stretch_args="--init $output_dir/output/${previous_basename}.npz --init-stretch-by $((previous_r/r))"
                    fi
                python gc_model_main.py -m $binned_mcool -ch ${chr} -kb ${r} -o $output_dir/output/${fit_basename}.npz -n $st --seed $i --iterations 1 --no-overwrite $stretch_args > $output_dir/logs/${fit_basename}.log 2>&1
                previous_basename=$fit_basename
                previous_r=$r
            done
        ) &
    done
    wait
    echo fraction=$s: Choosing best fits
    for r in ${resolutions[@]}; do
        fit_basename=${chr}_${st}st_s${s}_${r}
        python get_best.py $output_dir/output/${fit_basename}_run*.npz --target "$output_dir/output/${fit_basename}_best.npz"
    done
done
wait

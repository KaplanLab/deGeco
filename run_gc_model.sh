#!/bin/bash

#PBS -q S
#PBS -N tal_gc
#PBS -l nodes=1:ppn=1


#resolution 
r=100000
#mcool file 
m=/storage/md_kaplan/SharedData/Rao-Cell-2015/cool/Rao2014-GM12878-MboI-allreps-filtered.mcool
#m=/storage/md_kaplan/SharedData/Archive/HiC/Processed/Olfactory-Lomvardas-Nature-24Jan19/4DNFIUH9FJR6.mcool
#local - working directory
l=/storage/md_kaplan/haiak/Tal_model/

#name of new dir in local
d=hg19
mkdir -p ${l}/${d}
# creat a directory for output data
mkdir -p ${l}/${d}/OutputData
# creat a directory for output figures
mkdir -p ${l}/${d}/Figures

while read f; do

chr=`echo $f | awk '{split($0,a,":"); split(a[1],b,"r"); print b[2]}'`

qsub -V -v d1=$d,ch1=$chr,r1=$r,m1=$m,l1=$l -o /storage/md_kaplan/haiak/Tal_model -e /storage/md_kaplan/haiak/Tal_model -q P -N "tal-gc-"$chr -l nodes=1:ppn=1 -l mem=40gb -l vmem=40gb /storage/md_kaplan/haiak/Tal_model/Scripts/gc_model_script.sh

#qsub -V -v d1=$d,ch1=$chr,r1=$r,m1=$m,l1=$l -o /storage/md_kaplan/haiak/Tal_model -e /storage/md_kaplan/haiak/Tal_model -q P -N "tal-gc-"$chr -l nodes=1:ppn=1 -l mem=40gb -l vmem=40gb /storage/md_kaplan/haiak/Tal_model/Scripts/figures_script.sh

sleep 2

done < /storage/md_kaplan/haiak/GRID/hg19.chrome.size


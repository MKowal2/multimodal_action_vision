#!/bin/bash
indir="/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos/"
outdir="/mnt/zeta_share_1/m3kowal/videos/"
input="/home/m3kowal/Research/vfhlt/PyTorchConv3D/data/phav/videos/phavTrainTestlist/norainfog/data_list_full.txt"
IFS= ' '
while read vid class
do
  echo $(date)
  echo "${outdir}${vid}"
  rsync -rz "${indir}${vid}" "${outdir}${vid}"
#scp -r -C "${indir}${vid}" "${outdir}${vid}"
done < "$input"

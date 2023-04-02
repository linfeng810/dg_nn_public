#!/usr/bin/env bash

for mesh_no in {5..8..1}
do
  file=square_refine"${mesh_no}".msh
  nlevel=$((2*mesh_no+2))+1
	for ((level=0; level<nlevel; level++))
	do
		echo ${file} 'level' "${level}"
		echo output_${file}.out
		python3 main.py $file $level |& tee -a output_${file}_v33.out
		echo '==========='
	done
done

#!/usr/bin/env bash

for mesh_no in {8..3..-1}
do
  file=square_refine"${mesh_no}".msh
#  nlevel=$((2*mesh_no+2))+1
#	for ((level=0; level<nlevel; level++))
  for ((smoothstep=3; smoothstep>2; smoothstep--))
	do
		echo ${file} 'smooth step' "${smoothstep}"
		outfile=output_cubic_${file}_v${smoothstep}${smoothstep}.out
		echo '====memory issue solved version===' |& tee -a ${outfile}
		echo $'\n ====START-TIME====' |& tee -a ${outfile}
		date |& tee -a ${outfile}
		python3 main.py $file -1 $smoothstep |& tee -a ${outfile}
		date |& tee -a ${outfile}
		echo $'\n ====END-TIME====\n' |& tee -a ${outfile}
		echo '==========='
	done
done

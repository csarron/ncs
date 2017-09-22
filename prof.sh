#!/usr/bin/env bash
if [ "$1" == "-h" ]; then
  echo -e "\033[36mUsage: `basename $0` [net param file dir, e.g. nets/alexnet/gen/]"
  exit 0
fi

net=$1 #'nets/alexnet/gen/' 
echo "net dir: ${net}"
profile_dir="${net}/prof"
mkdir -p ${profile_dir}

file_list=(${net}/*.prototxt)
total=${#file_list[@]}
echo "total files: ${total}"
count=1
for proto in "${file_list[@]}";
do
    name=$(basename ${proto} ".prototxt")
    if [ ! -f "${net}/${name}.caffemodel"  ]; then
        echo "caffemodel not exists for ${proto}, skip"
        continue
    fi
    for i in $(seq 12);
    do
        echo -e "\x1b[1mprofiling (${count}/${total}) ${name}, using ${i} cores.\x1b[0m"
        profile_file="${profile_dir}/${name}_${i}.txt"
        if [ -f ${profile_file}  ]; then
            echo "${profile_file} already profiled, skip"
            continue
        fi
        python3 ../../bin/mvNCProfile.pyc ${proto} -s ${i} > ${profile_file}
        inference_time=$(grep 'Total inference time' ${profile_file} | awk  '{print $4}')
        if [[ ! -z ${inference_time} ]]; then
            report_file="${profile_dir}/${name}_${i}.html"
            echo "inference time: ${inference_time}"
            echo "saved report to ${report_file}"
            mv output_report.html ${report_file}
            graph_file="${profile_dir}/${name}_${i}.graph"
            echo "saved graph to ${graph_file}"
            mv graph ${graph_file}
        else
            echo "invalid graph, ${profile_file} removed"
            rm ${profile_file}
        fi
        echo
        echo
    done;

    ((count++))
done;
echo -e "\x1b[1mfinished"
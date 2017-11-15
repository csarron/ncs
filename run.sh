#!/usr/bin/env bash
if [ "$1" == "-h" ]; then
  echo -e "\033[36mUsage: `basename $0` [net param file dir, e.g. net_configs/alexnet/gen/]"
  exit 0
fi

net=$1 #'net_configs/alexnet/gen/'
echo "net dir: ${net}"
gen_dir="${net}/gen"
mkdir -p ${gen_dir}

file_list=(${net}/*.yaml)
total=${#file_list[@]}
echo "total files: ${total}"
count=1
for yaml_file in "${file_list[@]}";
do
    echo ${yaml_file}
    name=$(basename ${yaml_file} ".yaml")
    echo -e "\x1b[1mrunning (${count}/${total}) ${name}\x1b[0m"
    python gen_plain.py ${yaml_file}
    python see_net.py "${gen_dir}/${name}.prototxt" -s
    ((count++))
done;
echo -e "\x1b[1mfinished"
#!/usr/bin/env bash
if [ "$1" == "-h" ]; then
  echo -e "\033[36mUsage: `basename $0` [net param file, e.g. net.yaml]  [number of runs] "
  exit 0
fi

net=$1 #'nets/alexnet/alexnet.yaml'
nums=$2 
echo "net:"${net}
echo "nums:"${nums}

file_dir=$(dirname ${net})

for i in $(seq ${nums});
do
	python3 gen_net.py -p ${net} || true
	count=0
	for proto in `ls ${file_dir}/gen/*.prototxt`;
    do
        name=$(basename ${proto} ".prototxt")
        model=${name}".caffemodel"
        if [ ! -f ${file_dir}/gen/${model}  ]; then
            echo "caffemodel not exists, rm ${name}.prototxt";
            rm ${proto};
        else
            ((count++))
        fi
    done;
    echo
    echo -e "\x1b[1mrun: "${i}", current nets:"${count}"\x1b[0m"
    echo
done;

echo -e "\x1b[1mfinished"
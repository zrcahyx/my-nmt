#!/usr/bin/env bash

function print_help()
{
    if [[ 1 == $# ]]; then
        echo -e "\n\t\033[31m$1\033[0m"
    fi

    local usage="
    Usage: \n
    ./run.sh -h <host_name>

    host_name:
        guangzhou, wuhan, nanjing

    log file: train.log

    "
    echo -e "\033[33m${usage}\033[0m"
}

HOST="guangzhou"

if [ $# -eq 1 ]; then
    print_help
    exit 0
fi

while [[ $# > 1 ]]
do
key="$1"

case $key in
    -h|--hostname)
    HOST="$2"
    shift # past argument
    ;;
esac
shift # past argument or value
done

echo "#!/bin/bash">run.sh
echo "source /aifs/users/rcz56/env/bin/activate">>run.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64">>run.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64">>run.sh
echo "python train.py">>run.sh

# ROOT_DIR=`cd .. && pwd`
qsub -cwd -S /bin/bash -o train.log -j y -l hostname=$HOST -v PYTHONPATH=$ROOT_DIR run.sh

rm run.sh

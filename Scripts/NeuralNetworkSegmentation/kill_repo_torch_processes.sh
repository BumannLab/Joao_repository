#!/bin/bash

processes_id=$(nvidia-smi | grep Joao_repository | awk -F' ' '{print $5}')

for i in $processes_id
do
    kill -9 $i
done
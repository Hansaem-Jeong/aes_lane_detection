#!/bin/bash

first_dir="./data2/lane_data/ckpt/"
second_dir="result_"
dataset="kodasv1"
_view="all"
ckpt="laneatt_r18_culane"
data_dir="./data2/datasets/KODAS1/Input/"
conf_threshold="0.4"
nms_thres="45."
max_lane="2"

for i in $conf_threshold
do for j in $nms_thres
  do for k in $max_lane
    do CUDA_VISIBLE_DEVICES=0 python3 main.py test --exp_name ${ckpt} --view all --conf_threshold $i --nms_thres $j --max_lane $k --data_dir ${data_dir} --test_dataset ${dataset} --test_first_dir ${first_dir} --test_second_dir ${second_dir}
    done;
  done;
done;

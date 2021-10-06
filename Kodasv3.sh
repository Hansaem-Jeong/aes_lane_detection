#!/bin/bash

first_dir="/data2/lane_data/ckpt/"
second_dir="result_"
dataset="kodasv3"
_view="all"
ckpt="laneatt_r18_culane"
data_dir="/data2/datasets/KODAS_v3/"
conf_threshold="0.4"
nms_thres="40."
max_lane="2"

for i in $conf_threshold
do for j in $nms_thres
  do for k in $max_lane
    do CUDA_VISIBLE_DEVICES=3 python main.py test --exp_name ${ckpt} --view all --conf_threshold $i --nms_thres $j --max_lane $k --data_dir ${data_dir} --test_dataset ${dataset} --test_first_dir ${first_dir} --test_second_dir ${second_dir}
    done;
  done;
done;
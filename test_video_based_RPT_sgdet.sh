#!/usr/bin/env bash

dataset_size='large'
dataset_path='dataset/ag/'
train_model_name='model_image_based_RPT'
mode='sgdet'
cuda_visible_device='0'
device='cuda:0'


work_dir='work_dir/test/'$train_model_name'/'$mode
save_path=$work_dir
log_path=$work_dir/exp.log

model_path='ckpt/model_video_based_RPT_sgdet.tar'

if [[ ! -d "${work_dir}" ]]; then
  echo "${work_dir} doesn't exist. Creating it.";
  mkdir -p ${work_dir}
  python test_mini_batch.py -model_path $model_path -mode $mode -eval $eval -cuda_visible_device $cuda_visible_device -device $device -train_model_name $train_model_name -datasize $dataset_size -data_path $dataset_path -work_dir $work_dir -save_path $save_path -log_path $log_path
else
  echo "${work_dir} is existed."
  python test_mini_batch.py -model_path $model_path -mode $mode -eval $eval -cuda_visible_device $cuda_visible_device -device $device -train_model_name $train_model_name -datasize $dataset_size -data_path $dataset_path -work_dir $work_dir -save_path $save_path -log_path $log_path
fi
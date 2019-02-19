#!/bin/bash
# 训练集转化成 tfrecords存储的路径
DATASET_DIR=F:/DataSet/MyFace_5/MyFace_5/Save/
# 存储训练结果的路径，包括 checkpoint和event，自行指定
TRAIN_DIR=F:/DataSet/MyFace_5/MyFace_5/logs/
# 下载 vgg_16模型
# CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt

python ./train_ssd_network.py \
	 --train_dir=${TRAIN_DIR} \
	 --dataset_dir=${DATASET_DIR} \
	 --dataset_name=myface \
	 --dataset_split_name=train \
	 --model_name=ssd_300_vgg \
	 --save_summaries_secs=600 \
	 --save_interval_secs=600 \
	 --optimizer=adam \
	 --learning_rate_decay_factor=0.94 \
	 --batch_size=1 \
	 --gpu_memory_fraction=0.9
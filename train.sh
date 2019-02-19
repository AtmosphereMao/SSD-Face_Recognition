#!/bin/bash
# 训练集转化成 tfrecords存储的路径
DATASET_DIR=F:/DataSet/MyFace_4/MyFace_4/Save/
# 存储训练结果的路径，包括 checkpoint和event，自行指定
TRAIN_DIR=F:/DataSet/MyFace_4/MyFace_4/logs/
# 下载 vgg_16模型
CHECKPOINT_PATH=F:/Python文档/MachineLearning/SZPT/SSD-MyFace/checkpoints/vgg_16.ckpt

python ./train_ssd_network.py \
--train_dir=${TRAIN_DIR} \
--dataset_dir=${DATASET_DIR} \
--dataset_name=myface \
--dataset_split_name=train \
--model_name=ssd_300_vgg \
--save_summaries_secs=60 \
--save_interval_secs=600 \
--optimizer=adam \
--learning_rate=0.001 \
--learning_rate_decay_factor=0.94 \
--batch_size=1 \
--gpu_memory_fraction=0.9 \
--checkpoint_model_scope=vgg_16 \
--checkpoint_path=./checkpoints/vgg_16.ckpt \
--checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
--trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \

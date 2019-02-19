#!/bin/bash
#this is a shell script to convert pascal VOC datasets into tf-records only
#directory where the original dataset is stored
DATASET_DIR=F:/DataSet/MyFace_4/MyFace_4/
#VOC数据保存的文件夹（ 数据保存的文件夹（ 数据保存的文件夹（ 数据保存的文件夹（ 数据保存的文件夹（ VOC的目录 的目录 格式未改变） 格式未改变） 格式未改变）
#output directory where to store TFRecords files
OUTPUT_DIR=F:/DataSet/MyFace_4/MyFace_4/Save
#自己建立的保存 自己建立的保存 自己建立的保存 自己建立的保存 自己建立的保存 tfrecords数据的文件夹 数据的文件夹 数据的文件夹 数据的文件夹
python ./tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=${DATASET_DIR} --output_name=MyFace4_train --output_dir=${OUTPUT_DIR}
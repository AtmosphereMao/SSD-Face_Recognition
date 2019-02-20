SSD-Face_Recognition
 利用SSD做人脸预测
### 输入训练数据集到模型(ssd-tensorflow)进行训练

#### （1）修改训练类别 
修改 datasets 文件夹中 pascalvoc_common.py 文件  
<font color='red'>修改前</font>：
```python
VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
```
<font color='red'>修改后</font>：
```python
VOC_LABELS = {
    'none': (0, 'Background'),
    'person': (1, 'Person'),
}
```
##### （2）将图像数据（voc格式）转换为 tfrecods 格式 
修改 datasets 文件夹中 pascalvoc_to_tfrecords.py  文件  
更改文件的83行 读取模式和图片格式   
修改前：
```python
    filename = directory + DIRECTORY_IMAGES + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'r').read()
```
修改后：
```python
    filename = directory + DIRECTORY_IMAGES + name + '.png'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
```

在 SSD-Tensorflow-master 文件夹下创建 tf_conver_data.sh，并在目录下创建tfreocrds文件夹用于存储生成的tfrecords数据 
```
#!/usr/bin/env bash
#this is a shell script to convert pascal VOC datasets into tf-records only
#directory where the original dataset is stored
DATASET_DIR=/d/AIn/bisai/2/SSD-Tensorflow-master/SSD-Tensorflow-master/VOC/
#VOC 数据保存的文件夹（VOC 的目录格式未改变）
#output directory where to store TFRecords files
OUTPUT_DIR=/d/AIn/bisai/2/SSD-Tensorflow-master/SSD-Tensorflow-master/tfrecords/  # tfrecords 数据的文件夹
 python ./tf_convert_data.py \
        --dataset_name=pascalvoc \
        --dataset_dir=${DATASET_DIR} \
        --output_name=tfrecords_train \
        --output_dir=${OUTPUT_DIR}
```
在终端运行tf_conver_data.sh

#### （2）训练模型
##### 修改train_ssd_network.py 
最大训练步数（154行）：
```python
#None为无限训练步数，可修改为具体的数值如：50000
tf.app.flags.DEFINE_integer('max_number_of_steps', None, 
                            'The maximum number of training steps.')
```
模型保存间隔（59行）：
```python
#600秒保存一次模型
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
```
分割类别（134行）<font color='red'>修改前</font>：

```python
#根据自己的数据修改为类别+1 
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
```
分割类别（134行）<font color='red'>修改后</font>：
```python
#根据自己的数据修改为类别+1 
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
```

##### 修改nets/ssd_vgg_300.py
分割类别（96行）<font color='red'>修改前</font>：
```python
default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=21,     #根据自己的数据修改为类别+1 
        no_annotation_label=21, #根据自己的数据修改为类别+1 
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
```
分割类别（96行）<font color='red'>修改后</font>：
```python
default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=2,     #根据自己的数据修改为类别+1 
        no_annotation_label=2, #根据自己的数据修改为类别+1 
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        anchor_size_bounds=[0.15, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
```

##### 修改eval_ssd_network.py
分割类别（66行）<font color='red'>修改前</font>：
```python
#根据自己的数据修改为类别+1 
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
```
分割类别（66行）<font color='red'>修改后</font>：
```python
#根据自己的数据修改为类别+1 
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
```
##### 修改datasets/pascalvoc_2007.py
训练集，<font color='red'>修改前</font>：
```python
#各个类别（图片的数量，目标的数量）
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (238, 306),
    'bicycle': (243, 353),
    'bird': (330, 486),
    'boat': (181, 290),
    'bottle': (244, 505),
    'bus': (186, 229),
    'car': (713, 1250),
    'cat': (337, 376),
    'chair': (445, 798),
    'cow': (141, 259),
    'diningtable': (200, 215),
    'dog': (421, 510),
    'horse': (287, 362),
    'motorbike': (245, 339),
    'person': (2008, 4690),
    'pottedplant': (245, 514),
    'sheep': (96, 257),
    'sofa': (229, 248),
    'train': (261, 297),
    'tvmonitor': (256, 324),
    'total': (5011, 12608), #（图片的总量，目标的总量）
}
```
训练集，<font color='red'>修改后</font>：
```python
#各个类别（图片的数量，目标的数量）
TRAIN_STATISTICS = {
    'none': (0, 0),
    'person': (32, 32),
    'total': (32, 32), #（图片的总量，目标的总量）
}
```
测试集，<font color='red'>修改前</font>：
```python
#各个类别（图片的数量，目标的数量）
TEST_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (1, 1),
    'bicycle': (1, 1),
    'bird': (1, 1),
    'boat': (1, 1),
    'bottle': (1, 1),
    'bus': (1, 1),
    'car': (1, 1),
    'cat': (1, 1),
    'chair': (1, 1),
    'cow': (1, 1),
    'diningtable': (1, 1),
    'dog': (1, 1),
    'horse': (1, 1),
    'motorbike': (1, 1),
    'person': (1, 1),
    'pottedplant': (1, 1),
    'sheep': (1, 1),
    'sofa': (1, 1),
    'train': (1, 1),
    'tvmonitor': (1, 1),
    'total': (20, 20), #（图片的总量，目标的总量）
}
```
测试集，<font color='red'>修改后</font>：
```python
#各个类别（图片的数量，目标的数量）
TRAIN_STATISTICS = {
    'none': (0, 0),
    'person': (1, 1),
    'total': (1, 1), #（图片的总量，目标的总量）
}
```

<font color='red'>修改前</font>：
```python
SPLITS_TO_SIZES = {
    'train': 5011, #训练数据量
    'test': 4952,  #测试数据量
}
```
<font color='red'>修改后</font>：
```python
SPLITS_TO_SIZES = {
    'train': 32, #训练数据量
    'test': 1,  #测试数据量
}
```
<font color='red'>修改前</font>：
```python
NUM_CLASSES = 20 #根据自己的数据修改为类别数
```
<font color='red'>修改后</font>：
```python
NUM_CLASSES = 1 #根据自己的数据修改为类别数
```
新建train.sh文件

从头开始训练（如果没有vgg-16的checkpoints）
```
#!/bin/bash
# 训练集转化成 tfrecords存储的路径
DATASET_DIR=F:/DataSet/MyFace_3/MyFace_3/Save/
# 存储训练结果的路径，包括 checkpoint和event，自行指定
TRAIN_DIR=F:/DataSet/MyFace_3/MyFace_3/logs/

python ./train_ssd_network.py \
	 --train_dir=${TRAIN_DIR} \
	 --dataset_dir=${DATASET_DIR} \
	 --dataset_name=myface_2 \
	 --dataset_split_name=train \
	 --model_name=ssd_300_vgg \
	 --save_summaries_secs=600 \
	 --save_interval_secs=600 \
	 --optimizer=adam \
	 --learning_rate_decay_factor=0.94 \
	 --batch_size=5 \
	 --gpu_memory_fraction=0.9
```

基于 vgg-16 训练
```
DATASET_DIR=F:/DataSet/MyFace_3/MyFace_3/Save/
TRAIN_DIR=F:/DataSet/MyFace_3/MyFace_3/logs/
CHECKPOINT_PATH=./checkpoints/vgg_16.ckpt

python ./train_ssd_network.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=myface_2 \
    --dataset_split_name=train \
    --model_name=ssd_300_vgg \
    --save_summaries_secs=60 \
    --save_interval_secs=600 \
    --weight_decay=0.0005 \
    --optimizer=adam \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --batch_size=5 \
    --gpu_memory_fraction=0.9 \
    --checkpoint_model_scope=vgg_16 \
    --checkpoint_path=./checkpoints/vgg_16.ckpt \
    --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \
    --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box \

```

test 测试验证


```
ckpt_filename = 'F:/DataSet/MyFace_2/MyFace_2/logs/model.ckpt-5400' # 修改ckpt路径为自己的路径
```
```
# 自己修改的部分，上面都从notebook里复制黏贴
def CatchUsbVideo(window_name, camera_idx):
    # cv2.namedWindow(window_name)

    cap = cv2.VideoCapture(camera_idx)

    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            break

        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img = mpimg.imread(frame)
        rclasses, rscores, rbboxes = process_image(img)
        visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        cv2.imshow(window_name, img)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    CatchUsbVideo("video", 0)
```

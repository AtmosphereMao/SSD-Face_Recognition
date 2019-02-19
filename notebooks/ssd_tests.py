import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
import notebooks

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
## Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
# ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = 'F:/DataSet/MyFace_4/MyFace_4/logs/model.ckpt-120000'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.835, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess .run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape.        Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

def CatchUsbVideo(window_name, camera_idx):
    # cv2.namedWindow(window_name)

    cap = cv2.VidgiteoCapture(camera_idx)

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


# # Test on some demo image and visualize output.
# path = 'F:/DataSet/MyFace_3/MyFace_3/JPEGImages/'
# image_names = sorted(os.listdir(path))         # 获取测试文件夹所有图片
#
# i = 0
# for it  in image_names:
#     img = cv2.imread(path + it)
#     i+=1
#     if i>30: break                               # 测试4张图片
#     rclasses, rscores, rbboxes =  process_image(img)
#     visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
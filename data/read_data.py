import os
import numpy as np
import tensorflow as tf

from utils import FLAGS


def get_files(file_dir):
    D= []

    for (dirpath, dirnames, filenames) in os.walk(file_dir):
        for filename in filenames:
            D += [os.path.join(dirpath, filename)]

    temp = np.array([D])
    temp = temp.transpose()

    lists = list(temp[:,0])

    image_list = []

    for i in range(len(lists) - FLAGS.time_step):
        image_list.extend(lists[i:i+FLAGS.time_step])

    # print(image_list)
    return image_list


def get_batch_raw(image, image_H, image_W, batch_size,capacity):
    image = tf.cast(image, tf.string)

    #加入队列
    input_queue = tf.train.slice_input_producer([image], shuffle = False)

    #jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=1)

    image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)

    #对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)

    image_batch = tf.train.batch([image],batch_size = batch_size,num_threads=16,capacity = capacity)

    images_batch = tf.cast(image_batch, tf.float32)

    return images_batch


def get_label(start, label):

    label_list = np.zeros(shape=[FLAGS.batch_size, FLAGS.road_num])

    for i in range(FLAGS.batch_size):
        label_list[i] = label[start+i+FLAGS.time_step + FLAGS.pred_time - 1]

    return label_list

if __name__ == '__main__':
    image_list = get_files('E:/test/')
    # get_batch_raw(image_list,998,828,16,64)
    # get_label()




import tensorflow as tf
import numpy as np
# from tqdm import tqdm
import sys
import os
import time

from utils import FLAGS

'''tfrecord 写入数据.
将图片数据写入 tfrecord 文件。以 MNIST jpg格式数据集为例。

首先将图片解压到 ../../MNIST_data/mnist_jpg/ 目录下。
解压以后会有 training 和 testing 两个数据集。在每个数据集下，有十个文件夹，分别存放了这10个类别的数据。
每个文件夹名为对应的类别编码。

现在网上关于打包图片的例子非常多，实现方式各式各样，效率也相差非常多。
选择合适的方式能够有效地节省时间和硬盘空间。
有几点需要注意：
1.打包 tfrecord 的时候，千万不要使用 Image.open() 或者 matplotlib.image.imread() 等方式读取。
 1张小于10kb的jpg图片，前者（Image.open) 打开后，生成的对象100+kb, 后者直接生成 numpy 数组，大概是原图片的几百倍大小。
 所以应该直接使用 tf.gfile.FastGFile() 方式读入图片。
2.从 tfrecord 中取数据的时候，再用 tf.image.decode_jpg() 对图片进行解码。
3.不要随便使用 tf.image.resize_image_with_crop_or_pad 等函数，可以直接使用 tf.reshape()。前者速度极慢。
4.如果有固态硬盘的话，图片数据一定要放在固态硬盘中进行读取，速度能高几十倍几十倍几十倍！生成的 tfrecord 文件就无所谓了，找个机械盘放着就行。
'''

# jpg 文件路径
TRAINING_DIR = 'G:/SRCN/realDataFormat_800r_png'
# TESTING_DIR = '../../MNIST_data/mnist_jpg/testing/'
# tfrecord 文件保存路径,这里只保存一个 tfrecord 文件
TRAINING_TFRECORD_NAME = 'test.tfrecord'
# TESTING_TFRECORD_NAME = 'testing.tfrecord'


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))

def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[values]))


def convert_tfrecord_dataset(dataset_dir, tfrecord_name, tfrecord_path='../data/'):
    """ convert samples to tfrecord dataset.
    Args:
        dataset_dir: 数据集的路径。
        tfrecord_name: 保存为 tfrecord 文件名
        tfrecord_path: 保存 tfrecord 文件的路径。
    """
    if not os.path.exists(dataset_dir):
        print(u'jpg文件路径错误，请检查是否已经解压jpg文件。')
        exit()
    if not os.path.exists(os.path.dirname(tfrecord_path)):
        os.makedirs(os.path.dirname(tfrecord_path))

    label_list = np.genfromtxt('G:/SRCN/November_800r_velocity_cnn.txt')
    label_list = label_list.astype(int)
    # label_list = label_list.tolist()

    tfrecord_file = os.path.join(tfrecord_path, tfrecord_name)
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
            # 在该文件夹下，获取所有图片文件名
            file_names = os.listdir(dataset_dir)
            time0 = time.time()
            n_sample = len(file_names)
            # 共30630-12 = 30618，但是为了便于5 batch整除，所以减3 = 30615
            for i in range(54, 21441):
                # 每步步长为12
                for j in range(i, i+FLAGS.time_step):
                    file_name = file_names[j]
                    sys.stdout.write('\r>> Converting image %d/%d , %g s' % (
                        i + 1, n_sample, time.time() - time0))
                    jpg_path = os.path.join(dataset_dir, file_name)  # 获取每个图片的路径
                    # CNN inputs using
                    img = tf.gfile.FastGFile(jpg_path, 'rb').read()  # 读入图片
                    label = label_list[j+1]
                    # label.toString()
                    file = dict()
                    file['image'] = bytes_feature(img)
                    for index,velocity in enumerate(label):
                        file['label_{}'.format(index)] = int64_feature(velocity)
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature=file))
                    serialized = example.SerializeToString()
                    writer.write(serialized)
    print('\nFinished writing data to tfrecord files.')


if __name__ == '__main__':
    convert_tfrecord_dataset(TRAINING_DIR, TRAINING_TFRECORD_NAME)
    # convert_tfrecord_dataset(TESTING_DIR, TESTING_TFRECORD_NAME)
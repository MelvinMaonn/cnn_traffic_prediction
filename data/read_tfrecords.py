import tensorflow as tf

'''read data
从 tfrecord 文件中读取数据，对应数据的格式为png / jpg 等图片数据。
'''

# **1.把所有的 tfrecord 文件名列表写入队列中
filename_queue = tf.train.string_input_producer(['../data/second_ring_training.tfrecord'], num_epochs=None, shuffle=False)
# **2.创建一个读取器
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
# **3.根据你写入的格式对应说明读取的格式
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'image': tf.FixedLenFeature([], tf.string)
                                   }
                                   )
img = features['image']
# 这里需要对图片进行解码
img = tf.image.decode_png(img, channels=1)  # 这里，也可以解码为 1 通道
img = tf.reshape(img, [998, 828, 1])  # 28*28*3

print('img is', img)
# **4.通过 tf.train.shuffle_batch 或者 tf.train.batch 函数读取数据
"""
这里，你会发现每次取出来的数据都是一个类别的，除非你把 capacity 和 min_after_dequeue 设得很大，如
X_batch, y_batch = tf.train.shuffle_batch([img, label], batch_size=100,
                                          capacity=20000, min_after_dequeue=10000, num_threads=3)
这是因为在打包的时候都是一个类别一个类别的顺序打包的，所以每次填数据都是按照那个顺序填充进来。
只有当我们把队列容量舍得非常大，这样在队列中才会混杂各个类别的数据。但是这样非常不好，因为这样的话，
读取速度就会非常慢。所以解决方法是：
1.在写入数据的时候先进行数据 shuffle。
2.多存几个 tfrecord 文件，比如 64 个。
"""

X_batch = tf.train.batch([img], batch_size=24, capacity=100, num_threads=16)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# **5.启动队列进行数据读取
# 下面的 coord 是个线程协调器，把启动队列的时候加上线程协调器。
# 这样，在数据读取完毕以后，调用协调器把线程全部都关了。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
y_outputs = list()
for i in range(5):
    _X_batch = sess.run(X_batch)
    print('** batch %d' % i)
    print('_X_batch.shape:', tf.shape(_X_batch))

# **6.最后记得把队列关掉
coord.request_stop()
coord.join(threads)
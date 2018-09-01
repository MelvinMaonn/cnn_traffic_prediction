import tensorflow as tf

# class CNN(object):
#     def __init__(self):
#         pass

def conv2d(x, name, filter_size, in_channels, out_channels, strides):
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='W',
                                 shape=[filter_size, filter_size, in_channels, out_channels],
                                 dtype=tf.float32)  # tf.glorot_normal_initializer

        b = tf.get_variable(name='b',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())
        tf.summary.histogram(name=name + '/biases', values=b)

        con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

    return tf.nn.bias_add(con2d_op, b)


def batch_norm(name, x, mode):
    """Batch normalization."""
    with tf.variable_scope(name):
        x_bn = \
            tf.contrib.layers.batch_norm(
                inputs=x,
                decay=0.9,
                center=True,
                scale=True,
                epsilon=1e-5,
                updates_collections=None,
                is_training=mode == 'train',
                fused=True,
                data_format='NHWC',
                zero_debias_moving_mean=True,
                scope='BatchNorm'
            )

    return x_bn


def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


def max_pool(x, ksize, strides):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, strides, strides, 1],
                          padding='SAME',
                          name='max_pool')

from model import fcn
import model.cnn as CNN
from utils import FLAGS
import tensorflow as tf

import libs.metrics as metrics

class DCNN():

    def __init__(self, mode):

        self.mode = mode

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel], name='x_in')
            self.ys = tf.placeholder(tf.float32, [None, FLAGS.road_num], name='y_in')
            self.global_step = tf.placeholder(dtype=tf.int32, shape=[], name="global_step")
        with tf.name_scope('model'):
            self.buildmodel()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.create_training_op()


    def buildmodel(self):

        # 构建五层CNN
        #TODO these params can be changed
        filters = [12, 16, 32, 64, 64, 128]
        strides = [1, 2]

        feature_h = FLAGS.image_height
        feature_w = FLAGS.image_width

        count__ = 0
        min_size = min(FLAGS.image_height, FLAGS.image_width)
        while min_size > 1:
            min_size = (min_size + 1) // 2
            count__ += 1
        assert (FLAGS.cnn_count <= count__, "FLAGS.cnn_count should be <= {}!".format(count__))

        with tf.variable_scope('cnn'):
            x = self.xs
            x = tf.reshape(x, [-1, FLAGS.image_height, FLAGS.image_width, filters[0]])
            for i in range(FLAGS.cnn_count):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    x = CNN.conv2d(x=x, name='cnn-%d' % (i + 1), filter_size=3, in_channels=filters[i], out_channels=filters[i+1], strides=strides[0])
                    x = CNN.batch_norm('bn%d' % (i+1), x, self.mode)
                    x = CNN.leaky_relu(x, FLAGS.leakiness)
                    if i == 2 or i == 3:
                        continue
                    x = CNN.max_pool(x, 2, strides[1])

                    # _, feature_h, feature_w, _ = x.shape

        # TODO this dimension is computed by hand, and you can change the compute way by last line code
        feature_h = 42
        feature_w = 40

        # 构建Flatten
        with tf.name_scope('flatten'):
            cnn_flat = tf.reshape(x, [-1, feature_h*feature_w*filters[5]], name='flat')

        # 构建fully connected layer
        with tf.name_scope('fcn'):
            #TODO 需要确定大小
            W_fc1 = fcn.weight_variable([feature_h*feature_w*filters[5], FLAGS.road_num], name='W')
            tf.summary.histogram(name='fcn-1/weights', values=W_fc1)
            b_fc1 = fcn.bias_variable([FLAGS.road_num], name='b')
            tf.summary.histogram(name='fcn-1/biases', values=b_fc1)
            h_fc1 = tf.matmul(cnn_flat, W_fc1) + b_fc1

        # 构建Dropout
        # with tf.name_scope('dropout'):
        #     lstm_drop = tf.nn.dropout(self.lstm2.pred, keep_prob=0.2, name='drop')

        # 构建fully connected layer
        with tf.name_scope('fcn'):
            W_fc2 = fcn.weight_variable([FLAGS.road_num,FLAGS.road_num], name='W')
            tf.summary.histogram(name='fcn-2/weights', values=W_fc2)
            b_fc2 = fcn.bias_variable([FLAGS.road_num], name='b')
            tf.summary.histogram(name='fcn-2/biases', values=b_fc2)
            self.pred = tf.matmul(h_fc1, W_fc2) + b_fc2


    def compute_cost(self):
        self.losses = tf.reduce_mean(tf.abs(tf.reshape(self.pred,[-1]) - tf.reshape(self.ys,[-1])))
        tf.summary.scalar('loss', self.losses)

        preds = tf.reshape(self.pred, [-1])
        labels = tf.reshape(self.ys, [-1])
        self.rmse_train = metrics.masked_rmse_tf(preds, labels, 0)
        self.mae_train = metrics.masked_mae_tf(preds, labels, 0)
        self.mape_train = metrics.masked_mape_tf(preds, labels, 0)


    def create_training_op(self):

        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                              self.global_step,
                                              FLAGS.decay_steps,
                                              FLAGS.decay_rate,
                                              staircase=True)
        tf.summary.scalar('lrn_rate', self.lrn_rate)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                                          beta1=FLAGS.beta1,
                                          beta2=FLAGS.beta2).minimize(self.losses)


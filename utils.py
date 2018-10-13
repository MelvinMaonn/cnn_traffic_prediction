import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 330, 'image height')
tf.app.flags.DEFINE_integer('image_width', 319, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')
tf.app.flags.DEFINE_integer('road_num', 655, 'image channels as input')

tf.app.flags.DEFINE_integer('cnn_count', 5, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
# tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
# tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 20, 'maximum epochs')
# tf.app.flags.DEFINE_integer('batch_size', 40, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 1000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_integer('time_length', 1021, 'the time step')
tf.app.flags.DEFINE_integer('time_step', 12, 'the time step')
tf.app.flags.DEFINE_integer('pred_time', 1, 'the time step')
tf.app.flags.DEFINE_integer('input_size', 655, 'the time step')
tf.app.flags.DEFINE_integer('output_size', 655, 'the time step')
tf.app.flags.DEFINE_integer('cell_size', 1024, 'the time step')
tf.app.flags.DEFINE_integer('batch_size', 5, 'the time step')

tf.app.flags.DEFINE_string('train_dir', './imgs/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './imgs/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './imgs/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')


import os
import numpy as np
from datetime import datetime

def make_dirlist(dirlist):
    for dir in dirlist:
        if not os.path.exists(dir):
            os.makedirs(dir)

time_fmt = "%Y-%m-%d-%H-%M-%S"

def now2string(fmt="%Y-%m-%d-%H-%M-%S"):
    return datetime.now().strftime(fmt)

def mape(pred, target):
    return np.abs(pred - target) / target

global_start_time = now2string()

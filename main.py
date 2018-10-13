import datetime

import tensorflow as tf
import numpy as np
import time

from model.srcn import SRCN
import data.read_data as rdd
from utils import *

from log import *


import os
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

logger = logging.getLogger('Train for CNN')
logger.setLevel(logging.INFO)

def train(mode='train'):
    srcn = SRCN(mode)
    srcn.buildmodel()
    srcn.compute_cost()
    global_step = tf.train.get_or_create_global_step()

    lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                          global_step,
                                          FLAGS.decay_steps,
                                          FLAGS.decay_rate,
                                          staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=lrn_rate,
                                      beta1=FLAGS.beta1,
                                      beta2=FLAGS.beta2).minimize(srcn.losses,
                                                                  global_step=global_step)

    num_train_samples = 1021 * 27
    num_batches_train_per_epoch = int((num_train_samples - FLAGS.time_step) / FLAGS.batch_size)
    num_val_samples = 1021 * 3
    num_batches_val_per_epoch = int((num_val_samples - FLAGS.time_step) / FLAGS.batch_size)

    tr_image = rdd.get_files('/mnt/data5/mm/data/traffic/train_2/')
    train_img = rdd.get_batch_raw(tr_image, FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step,100)
    va_image = rdd.get_files('/mnt/data5/mm/data/traffic/test/')
    val_image = rdd.get_batch_raw(va_image, FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step,100)
    train_label = np.genfromtxt('data/800r_train_2.txt')
    val_label = np.genfromtxt('data/800r_test.txt')

    logger_val = open("logdir/" + global_start_time + "_val.txt", 'w')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
        #train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from checkpoint{0}'.format(ckpt))


        print('=============================begin training=============================')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for cur_epoch in range(FLAGS.num_epochs):
                start_time = time.time()

                all_train_loss = np.zeros(4)
                all_val_loss = np.zeros(4)


                # the training part
                for cur_batch in range(num_batches_train_per_epoch):

                    x = sess.run(train_img)
                    y = rdd.get_label(cur_batch*FLAGS.batch_size, train_label)

                    feed_dict = {
                        srcn.xs: x,
                        srcn.ys: y,
                    }

                    results = sess.run([train_op, srcn.pred, srcn.losses, srcn.rmse_train, srcn.mae_train, srcn.mape_train],
                                       feed_dict=feed_dict)

                    all_train_loss += np.array(results[2:])

                    tf.summary.scalar('lrn_rate', lrn_rate)

                    if cur_batch % 100 == 0 and cur_batch != 0:
                        rs = sess.run(merged, feed_dict=feed_dict)
                        writer.add_summary(rs, cur_batch)
                        print(str(cur_batch) + ":" + str(all_train_loss / cur_batch + 1))

                # save the checkpoint
                if not os.path.isdir(FLAGS.checkpoint_dir):
                    os.mkdir(FLAGS.checkpoint_dir)
                    print('no checkpoint')
                logger.info('save checkpoint at step {0}', format(cur_epoch))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'srcn-model.ckpt'), global_step=cur_epoch)

                # the val part
                for cur_batch in range(num_batches_val_per_epoch):

                    x = sess.run(val_image)
                    y = rdd.get_label(cur_batch * FLAGS.batch_size, val_label)

                    feed_dict = {
                        srcn.xs: x,
                        srcn.ys: y,
                    }

                    results = sess.run([srcn.pred, srcn.losses, srcn.rmse_train, srcn.mae_train, srcn.mape_train],
                                       feed_dict=feed_dict)

                    all_val_loss += np.array(results[1:])

                    if cur_batch % 100 == 0:
                        print(str(cur_batch) + ":" + str(all_val_loss / cur_batch + 1))

                now = datetime.now()
                log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                      "train_loss = {}, " \
                      "val_loss = {}, " \
                      "time = {:.3f}"
                log = log.format(now.month, now.day, now.hour, now.minute, now.second,
                                 cur_epoch + 1, FLAGS.num_epochs,str(all_train_loss / num_batches_train_per_epoch),
                                 str(all_val_loss / num_batches_val_per_epoch), time.time() - start_time)
                print(log)

                logger_val.write(log+'\r\n')

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()
            logger_val.close()

        coord.join(threads=threads)

def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


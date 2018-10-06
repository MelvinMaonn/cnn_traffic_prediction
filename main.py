import datetime

import tensorflow as tf
import numpy as np
import time

from model.srcn import SRCN
import data.read_data as rdd
from utils import FLAGS



import os
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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

    num_train_samples = 1021 * 21
    num_batches_per_epoch = int((num_train_samples - 12) / FLAGS.batch_size)
    num_val_samples = 1021 * 6
    num_batches_val = int(num_val_samples / FLAGS.batch_size)

    tr_image = rdd.get_files('/mnt/data1/mm/SRCN/data/data_30min/train/')
    #train_img = rdd.get_batch_raw(tr_image, FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step,100)
    va_image = rdd.get_files('/mnt/data1/mm/SRCN/data/data_30min/val/')
    val_image = rdd.get_batch_raw(va_image, FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step,100)
    #train_label = np.genfromtxt('data/800r_train.txt')
    val_label = np.genfromtxt('data/800r_train.txt')

    f = open('train_and_val_result.txt', 'w')

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

                '''
                # the training part
                for cur_batch in range(num_batches_per_epoch):

                    #x, y = sess.run([train_image, train_label])
                    x = sess.run(train_img)
                    y = rdd.get_label(cur_batch*FLAGS.batch_size, train_label)

                    feed_dict = {
                        srcn.xs: x,
                        srcn.ys: y,
                        # create initial state
                    }

                    _, loss, pred = sess.run([train_op, srcn.losses, srcn.pred], feed_dict=feed_dict)
                    # calculate the cost
                    #train_cost += batch_cost * FLAGS.batch_size

                    tf.summary.scalar('lrn_rate', lrn_rate)

                    if cur_batch % 100 == 0:
                        rs = sess.run(merged, feed_dict=feed_dict)
                        writer.add_summary(rs, cur_batch)
                        print(str(cur_batch) + ":" + str(loss))

                # save the checkpoint
                if not os.path.isdir(FLAGS.checkpoint_dir):
                    os.mkdir(FLAGS.checkpoint_dir)
                    print('no checkpoint')
                logger.info('save checkpoint at step {0}', format(cur_epoch))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'srcn-model.ckpt'), global_step=cur_epoch)
                '''

                for cur_batch in range(num_batches_val):

                    x = sess.run(val_image)
                    y = rdd.get_label(cur_batch * FLAGS.batch_size, val_label)

                    feed_dict = {
                        srcn.xs: x,
                        srcn.ys: y,
                    }

                    loss, pred = sess.run([srcn.losses, srcn.pred], feed_dict=feed_dict)

                    if cur_batch % 100 == 0:
                        print(loss)


                now = datetime.datetime.now()
                log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                      "loss = {:.3f}, " \
                      "time = {:.3f}"
                log = log.format(now.month, now.day, now.hour, now.minute, now.second,
                                 cur_epoch + 1, FLAGS.num_epochs, loss, time.time() - start_time)
                print(log)

                f.write(log+'\r\n')

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()
            f.close()

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

'''
if __name__ == '__main__':

    srcn = SRCN('train')
    srcn.buildmodel()
    srcn.compute_cost()
    global_step = tf.train.get_or_create_global_step()
    lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                          global_step,
                                          FLAGS.decay_steps,
                                          FLAGS.decay_rate,
                                          staircase=True)
    # train_op = tf.train.RMSPropOptimizer(learning_rate=lrn_rate, momentum=FLAGS.momentum).minimize(srcn.cost,global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate=lrn_rate,
                                      beta1=FLAGS.beta1,
                                      beta2=FLAGS.beta2).minimize(srcn.losses,
                                                                  global_step=global_step)

    x = rdd.get_batch2(FLAGS.image_height, FLAGS.image_width, FLAGS.batch_size * FLAGS.time_step, 100)

    label = np.genfromtxt('drive/SRCN/November_800r_velocity_cnn.txt')
    # label = np.genfromtxt('E:/data/Data0/output4/SRCN/November_800r_velocity_cnn.txt')

    saver = tf.train.Saver()

    with tf.Session() as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("drive/CNN/logs", sess.graph)
        # writer = tf.summary.FileWriter("logs", sess.graph)


        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            for i in range(((int)((30630 - 12 - 3) / 5)) * 20):

                if coord.should_stop():
                    break

                if i == 0:
                    x = x.eval()

                y = rdd.get_label(i, label)

                feed_dict = {
                    srcn.xs: x,
                    srcn.ys: y,
                }

                _, loss, pred = sess.run([train_op, srcn.losses, srcn.pred], feed_dict=feed_dict)

                for j in range(5):
                    print(pred[j * 12 + 11][144])

                tf.summary.scalar('lrn_rate', lrn_rate)

                if i % 100 == 0:
                    rs = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(rs, i)
                    saver.save(sess, "drive/CNN/my_net_10/save_net.ckpt")

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads=threads)
'''

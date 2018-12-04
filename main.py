import datetime
import logging
import os
import time

import numpy as np

import data.read_data as rdd
from config import NUM_BATCHES_TRAIN_PER_EPOCH, NUM_BATCHES_VAL_PER_EPOCH, TRAIN_IMAGE_DIR, TRAIN_LABEL_PATH, \
    VAL_IMAGE_DIR, VAL_LABEL_PATH
from model.dcnn import DCNN
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

logger = logging.getLogger('Train for CNN')
logger.setLevel(logging.INFO)

def train(mode='train'):

    dcnn = DCNN(mode)

    train_image, train_label = get_image_and_label(TRAIN_IMAGE_DIR, TRAIN_LABEL_PATH)
    val_image, val_label = get_image_and_label(VAL_IMAGE_DIR, VAL_LABEL_PATH)

    logger_val = open("logdir/" + global_start_time + "_val.txt", 'w')

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)
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
                for cur_batch in range(NUM_BATCHES_TRAIN_PER_EPOCH):

                    global_step = cur_batch + cur_epoch * cur_batch

                    x = sess.run(train_image)
                    y = rdd.get_label(cur_batch*FLAGS.batch_size, train_label)

                    feed_dict = {
                        dcnn.xs: x,
                        dcnn.ys: y,
                        dcnn.global_step: global_step
                    }

                    results = sess.run([dcnn.train_op, dcnn.pred, dcnn.losses, dcnn.rmse_train, dcnn.mae_train, dcnn.mape_train],
                                       feed_dict=feed_dict)

                    all_train_loss += np.array(results[2:])

                    if cur_batch % 100 == 0 and cur_batch != 0:
                        rs = sess.run(merged, feed_dict=feed_dict)
                        writer.add_summary(rs, cur_batch)
                        print(str(cur_batch) + ":" + str(all_train_loss / cur_batch + 1))

                # save the checkpoint
                if not os.path.isdir(FLAGS.checkpoint_dir):
                    os.mkdir(FLAGS.checkpoint_dir)
                    print('no checkpoint')
                logger.info('save checkpoint at step {0}', format(cur_epoch))
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'dcnn-model.ckpt'), global_step=cur_epoch)

                # the val part
                for cur_batch in range(NUM_BATCHES_VAL_PER_EPOCH):

                    x = sess.run(val_image)
                    y = rdd.get_label(cur_batch * FLAGS.batch_size, val_label)

                    feed_dict = {
                        dcnn.xs: x,
                        dcnn.ys: y,
                    }

                    results = sess.run([dcnn.pred, dcnn.losses, dcnn.rmse_train, dcnn.mae_train, dcnn.mape_train],
                                       feed_dict=feed_dict)

                    all_val_loss += np.array(results[1:])

                    if cur_batch % 100 == 0 and cur_batch != 0:
                        print(str(cur_batch) + ":" + str(all_val_loss / cur_batch + 1))

                now = datetime.now()
                log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                      "train_loss = {}, " \
                      "val_loss = {}, " \
                      "time = {:.3f}"
                log = log.format(now.month, now.day, now.hour, now.minute, now.second,
                                 cur_epoch + 1, FLAGS.num_epochs,str(all_train_loss / NUM_BATCHES_TRAIN_PER_EPOCH),
                                 str(all_val_loss / NUM_BATCHES_VAL_PER_EPOCH), time.time() - start_time)
                print(log)

                logger_val.write(log+'\r\n')

        except tf.errors.OutOfRangeError:
            print('Done training epoch limit reached')
        finally:
            coord.request_stop()
            logger_val.close()

        coord.join(threads=threads)


def get_image_and_label(image_dir, label_path):
    image_path = rdd.get_files(image_dir)
    image = rdd.get_batch_raw(image_path, FLAGS.image_height, FLAGS.image_width,
                                    FLAGS.batch_size * FLAGS.time_step, 100)
    label = np.genfromtxt(label_path)
    return image, label


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


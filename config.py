from utils import FLAGS

NUM_TRAIN_SAMPLES = 1021 * 27
NUM_BATCHES_TRAIN_PER_EPOCH = int((NUM_TRAIN_SAMPLES - FLAGS.time_step - FLAGS.pred_time + 1) / FLAGS.batch_size)

NUM_VAL_SAMPLES = 1021 * 3
NUM_BATCHES_VAL_PER_EPOCH = int((NUM_VAL_SAMPLES - FLAGS.time_step - FLAGS.pred_time + 1) / FLAGS.batch_size)


TRAIN_IMAGE_DIR = '/mnt/data1/zll/mm/data/train_2/'
TRAIN_LABEL_PATH = 'data/800r_train_2.txt'

VAL_IMAGE_DIR = '/mnt/data1/zll/mm/data/test/'
VAL_LABEL_PATH = 'data/800r_test.txt'
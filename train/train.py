#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import abspath, dirname, join
import sys
import tensorflow as tf
from model import NMTModel
from helpers.misc_helper import create_hparams
from helpers.input_helper import get_infer_input, get_input
from helpers.vocab_helper import create_idx2vocab_tables, create_vocab2idx_tables

sys.path.append(dirname(dirname(abspath(__file__))))
from utils.config_helper import ConfigHelper

flags = tf.flags
logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

CHECKPOINT_BASENAME = 'model.ckpt'

cfg = ConfigHelper()
hparams = create_hparams(cfg)

def _run_training():
    data_dir = join(dirname(abspath(__file__)), 'data')
    # vocab files
    src_vocab_file = join(data_dir, hparams.vocab_prefix + '.' + hparams.src)
    tgt_vocab_file = join(data_dir, hparams.vocab_prefix + '.' + hparams.tgt)
    src_vocab2idx_table, tgt_vocab2idx_table = create_vocab2idx_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    src_idx2vocab_table, tgt_idx2vocab_table = create_idx2vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    # train files
    train_src_file = join(data_dir, hparams.train_prefix + '.' + hparams.src)
    train_tgt_file = join(data_dir, hparams.train_prefix + '.' + hparams.tgt)
    with tf.name_scope('TrainInput'):
        train_inputs = get_input(
            hparams, tf.contrib.learn.ModeKeys.TRAIN, train_src_file,
            train_tgt_file, src_vocab2idx_table, tgt_vocab2idx_table
            )
    # dev files
    dev_src_file = join(data_dir, hparams.dev_prefix + '.' + hparams.src)
    dev_tgt_file = join(data_dir, hparams.dev_prefix + '.' + hparams.tgt)
    with tf.name_scope('DevInput'):
        dev_inputs = get_input(
            hparams, tf.contrib.learn.ModeKeys.EVAL, dev_src_file,
            dev_tgt_file, src_vocab2idx_table, tgt_vocab2idx_table
            )
    # test files
    test_src_file = join(data_dir, hparams.test_prefix + '.' + hparams.src)
    test_tgt_file = join(data_dir, hparams.test_prefix + '.' + hparams.tgt)
    with tf.name_scope('TestInput'):
        test_inputs = get_infer_input(
            hparams, test_src_file, src_vocab2idx_table
        )

    # with tf.device('/gpu:{}'.format(hparams.gpu_list[0])):
    with tf.name_scope('Train'):
        with tf.variable_scope('NMTModel', reuse=None):
            train_model = NMTModel(
                hparams,
                tf.contrib.learn.ModeKeys.TRAIN,
                train_inputs,
                src_vocab2idx_table,
                tgt_vocab2idx_table
            )

    with tf.name_scope('Dev'):
        with tf.variable_scope('NMTModel', reuse=True):
            dev_model = NMTModel(
                hparams,
                tf.contrib.learn.ModeKeys.EVAL,
                dev_inputs,
                src_vocab2idx_table,
                tgt_vocab2idx_table
            )

    with tf.name_scope('Test'):
        with tf.variable_scope('NMTModel', reuse=True):
            test_model = NMTModel(
                hparams,
                tf.contrib.learn.ModeKeys.INFER,
                test_inputs,
                src_vocab2idx_table,
                tgt_vocab2idx_table
            )

    print('Graph build successfully!')
    input('> ')

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    sv = tf.train.Supervisor(
        logdir=hparams.save_path, save_model_secs=0, save_summaries_secs=3000)
    with sv.managed_session(config=sess_config) as sess:
        # random, infinite
        sess.run(dev_inputs.initializer)
        # batch, 1 epoch
        sess.run(test_inputs.initializer)
        min_dev_loss = float('inf')
        for epoch in xrange(hparams.num_train_epochs):
            sess.run(train_inputs.initializer)
            step = -1
            while True:
                step += 1
                try:
                    _, train_loss, train_sample_ids = sess.run([
                        train_model.train_op, train_model.loss,
                        train_model.sample_ids
                    ])
                    if epoch == 0 and step == 0:
                        avg_loss = train_loss
                    else:
                        avg_loss = 0.9 * avg_loss + 0.1 * train_loss
                    logging.info('training loss for epoch {} step {} is {}'.format(
                        epoch + 1, step + 1, avg_loss
                    ))
                    if step % hparams.steps_per_stats == 0:
                        dev_loss, dev_sample_ids = sess.run([
                            dev_model.loss, dev_model.sample_ids
                        ])
                        if epoch == 0 and step == 0:
                            avg_dev_loss = dev_loss
                        else:
                            avg_dev_loss = 0.9 * avg_dev_loss + 0.1 * dev_loss
                        logging.info(
                            'train loss for epoch {} is {}'.format(
                                epoch + 1, avg_loss
                        ))
                        logging.info(
                            'dev loss for epoch {} is {}'.format(
                                epoch + 1, avg_dev_loss
                            ))
                        if avg_dev_loss < min_dev_loss:
                            min_dev_loss = avg_dev_loss
                            sv.saver.save(
                                sess, join(hparams.save_path, CHECKPOINT_BASENAME))
                except tf.errors.OutOfRangeError:
                    break

        sv.saver.restore(sess, tf.train.latest_checkpoint(hparams.save_path))
        infer_ids = sess.run(test_model.infer_ids)


def main(_):
    _run_training()


if __name__ == '__main__':
    tf.app.run()

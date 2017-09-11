#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import abspath, dirname, join
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

sys.path.append(dirname(dirname(abspath(__file__))))
from train.model import NMTModel
from train.helpers.misc_helper import create_hparams
from train.helpers.input_helper import get_infer_input, get_input
from train.helpers.vocab_helper import create_idx2vocab_tables, create_vocab2idx_tables
from train.helpers.evaluation_helper import compute_bleu
from utils.config_helper import ConfigHelper

logging = tf.logging
logging.set_verbosity(tf.logging.INFO)

cfg = ConfigHelper()
hparams = create_hparams(cfg)

CHECKPOINT_BASENAME = 'model.ckpt'


def _export_model():
    test_src_file = tf.placeholder(tf.string, [None])
    data_dir = join(dirname(dirname(abspath(__file__))), 'train', 'data')
    # vocab files
    src_vocab_file = join(data_dir, hparams.vocab_prefix + '.' + hparams.src)
    tgt_vocab_file = join(data_dir, hparams.vocab_prefix + '.' + hparams.tgt)
    src_vocab2idx_table, tgt_vocab2idx_table = create_vocab2idx_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)
    src_idx2vocab_table, tgt_idx2vocab_table = create_idx2vocab_tables(
        src_vocab_file, tgt_vocab_file, hparams.share_vocab)

    with tf.name_scope('TestInput'):
        test_inputs = get_infer_input(
            hparams, test_src_file, src_vocab2idx_table
        )

    with tf.name_scope('Test'):
        with tf.variable_scope('NMTModel', reuse=None):
            test_model = NMTModel(
                hparams,
                tf.contrib.learn.ModeKeys.INFER,
                test_inputs,
                src_vocab2idx_table,
                tgt_vocab2idx_table,
                src_idx2vocab_table,
                tgt_idx2vocab_table
            )

    # restore checkpoint
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(hparams.save_path))

    model_signature = signature_def_utils.build_signature_def(
        inputs={'inputs': utils.build_tensor_info(test_src_file)},
        outputs={'bpe_sentences': utils.build_tensor_info(test_model.infer_bpe_sentences)},
        method_name=signature_constants.PREDICT_METHOD_NAME)

    builder = saved_model_builder.SavedModelBuilder(hparams.export_path)
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            model_signature,
        },
        legacy_init_op=tf.group(
            tf.initialize_all_tables(), name='legacy_init_op'))

    builder.save()


def main(_):
    _export_model()


if __name__ == '__main__':
    tf.app.run()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generally useful utility functions."""
from __future__ import print_function

# import codecs
import collections
# import json
# import math
# import os
# import sys
# import time

# import numpy as np
import tensorflow as tf
# import vocab_helper


# def check_tensorflow_version():
#   if tf.__version__ < "1.2.1":
#     raise EnvironmentError("Tensorflow version must >= 1.2.1")


# def safe_exp(value):
#   """Exponentiation with catching of overflow error."""
#   try:
#     ans = math.exp(value)
#   except OverflowError:
#     ans = float("inf")
#   return ans


# def print_time(s, start_time):
#   """Take a start time, print elapsed duration, and return a new time."""
#   print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
#   sys.stdout.flush()
#   return time.time()


# def print_out(s, f=None, new_line=True):
#   """Similar to print but with support to flush and output to a file."""
#   if isinstance(s, bytes):
#     s = s.decode("utf-8")

#   if f:
#     f.write(s.encode("utf-8"))
#     if new_line:
#       f.write(b"\n")

#   # stdout
#   out_s = s.encode("utf-8")
#   if not isinstance(out_s, str):
#     out_s = out_s.decode("utf-8")
#   print(out_s, end="", file=sys.stdout)

#   if new_line:
#     sys.stdout.write("\n")
#   sys.stdout.flush()


# def print_hparams(hparams, skip_patterns=None):
#   """Print hparams, can skip keys based on pattern."""
#   values = hparams.values()
#   for key in sorted(values.keys()):
#     if not skip_patterns or all(
#         [skip_pattern not in key for skip_pattern in skip_patterns]):
#       print_out("  %s=%s" % (key, str(values[key])))


# def load_hparams(model_dir):
#   """Load hparams from an existing model directory."""
#   hparams_file = os.path.join(model_dir, "hparams")
#   if tf.gfile.Exists(hparams_file):
#     print_out("# Loading hparams from %s" % hparams_file)
#     with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
#       try:
#         hparams_values = json.load(f)
#         hparams = tf.contrib.training.HParams(**hparams_values)
#       except ValueError:
#         print_out("  can't load hparams file")
#         return None
#     return hparams
#   else:
#     return None


# def maybe_parse_standard_hparams(hparams, hparams_path):
#   """Override hparams values with existing standard hparams config."""
#   if not hparams_path:
#     return hparams

#   if tf.gfile.Exists(hparams_path):
#     print_out("# Loading standard hparams from %s" % hparams_path)
#     with tf.gfile.GFile(hparams_path, "r") as f:
#       hparams.parse_json(f.read())

#   return hparams


# def save_hparams(out_dir, hparams):
#   """Save hparams."""
#   hparams_file = os.path.join(out_dir, "hparams")
#   print_out("  saving hparams to %s" % hparams_file)
#   with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
#     f.write(hparams.to_json())


# def debug_tensor(s, msg=None, summarize=10):
#   """Print the shape and value of a tensor at test time. Return a new tensor."""
#   if not msg:
#     msg = s.name
#   return tf.Print(s, [tf.shape(s), s], msg + " ", summarize=summarize)


# def add_summary(summary_writer, global_step, tag, value):
#   """Add a new summary to the current summary_writer.
#   Useful to log things that are not part of the training graph, e.g., tag=BLEU.
#   """
#   summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#   summary_writer.add_summary(summary, global_step)


# def get_config_proto(log_device_placement=False, allow_soft_placement=True):
#   # GPU options:
#   # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
#   config_proto = tf.ConfigProto(
#       log_device_placement=log_device_placement,
#       allow_soft_placement=allow_soft_placement)
#   config_proto.gpu_options.allow_growth = True
#   return config_proto


def format_text(words):
  """Convert a sequence words into sentence."""
  if (not hasattr(words, "__len__") and  # for numpy array
      not isinstance(words, collections.Iterable)):
    words = [words]
  return b" ".join(words)


def format_bpe_text(symbols, delimiter=b"@@"):
  """Convert a sequence of bpe words into sentence."""
  words = []
  word = b""
  if isinstance(symbols, str):
    symbols = symbols.encode()
  delimiter_len = len(delimiter)
  for symbol in symbols:
    if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
      word += symbol[:-delimiter_len]
    else:  # end of a word
      word += symbol
      words.append(word)
      word = b""
  return b" ".join(words)


def create_hparams(cfg):
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      src=cfg.get_string_value('Data', 'src'),
      tgt=cfg.get_string_value('Data', 'tgt'),
      train_prefix=cfg.get_string_value('Data', 'train_prefix'),
      dev_prefix=cfg.get_string_value('Data', 'dev_prefix'),
      test_prefix=cfg.get_string_value('Data', 'test_prefix'),
      vocab_prefix=cfg.get_string_value('Data', 'vocab_prefix'),
      out_dir=cfg.get_string_value('Data', 'out_dir'),

      # Networks
      num_units=cfg.get_int_value('Networks', 'num_units'),
      num_layers=cfg.get_int_value('Networks', 'num_layers'),
      dropout=cfg.get_float_value('Networks', 'dropout'),
      unit_type=cfg.get_string_value('Networks', 'unit_type'),
      encoder_type=cfg.get_string_value('Networks', 'encoder_type'),
      residual=cfg.get_boolean_value('Networks', 'residual'),
      time_major=cfg.get_boolean_value('Networks', 'time_major'),
      num_embeddings_partitions=cfg.get_int_value('Networks',
                                                  'num_embeddings_partitions'),

      # Attention mechanisms
      attention=cfg.get_string_value('Attention', 'attention'),
      attention_architecture=cfg.get_string_value('Attention',
                                                  'attention_architecture'),
      pass_hidden_state=cfg.get_boolean_value('Attention', 'pass_hidden_state'),

      # Train
      optimizer=cfg.get_string_value('Train', 'optimizer'),
      num_train_epochs=cfg.get_int_value('Train', 'num_train_epochs'),
      num_train_steps=cfg.get_int_value('Train', 'num_train_steps'),
      batch_size=cfg.get_int_value('Train', 'batch_size'),
      init_op=cfg.get_string_value('Train', 'init_op'),
      init_weight=cfg.get_float_value('Train', 'init_weight'),
      max_gradient_norm=cfg.get_float_value('Train', 'max_gradient_norm'),
      learning_rate=cfg.get_float_value('Train', 'learning_rate'),
      start_decay_step=cfg.get_int_value('Train', 'start_decay_step'),
      decay_factor=cfg.get_float_value('Train', 'decay_factor'),
      decay_steps=cfg.get_int_value('Train', 'decay_steps'),
      colocate_gradients_with_ops=cfg.get_string_value(
                                            'Train',
                                            'colocate_gradients_with_ops'),

      # Data constraints
      num_buckets=cfg.get_int_value('Data_constraints', 'num_buckets'),
      max_train=cfg.get_int_value('Data_constraints', 'max_train'),
      src_max_len=cfg.get_int_value('Data_constraints', 'src_max_len'),
      tgt_max_len=cfg.get_int_value('Data_constraints', 'tgt_max_len'),
      source_reverse=cfg.get_boolean_value('Data_constraints', 'source_reverse'),

      # Inference
      src_max_len_infer=cfg.get_int_value('Inference', 'src_max_len_infer'),
      tgt_max_len_infer=cfg.get_int_value('Inference', 'tgt_max_len_infer'),
      infer_batch_size=cfg.get_int_value('Inference', 'infer_batch_size'),
      beam_width=cfg.get_int_value('Inference', 'beam_width'),
      length_penalty_weight=cfg.get_float_value('Inference',
                                                'length_penalty_weight'),

      # Vocab
      sos=(cfg.get_string_value('Vocab', 'sos')
           if cfg.get_string_value('Vocab', 'sos') else vocab_helper.SOS),
      eos=(cfg.get_string_value('Vocab', 'eos')
           if cfg.get_string_value('Vocab', 'sos') else vocab_helper.EOS),
      bpe_delimiter=cfg.get_string_value('Vocab', 'bpe_delimiter'),
      src_vocab_size = cfg.get_int_value('Vocab', 'src_vocab_size'),
      tgt_vocab_size = cfg.get_int_value('Vocab', 'tgt_vocab_size'),
      src_embed_size = cfg.get_int_value('Vocab', 'src_embed_size'),
      tgt_embed_size = cfg.get_int_value('Vocab', 'tgt_embed_size'),


      # Misc
      forget_bias=cfg.get_float_value('Misc', 'forget_bias'),
      gpu_list=cfg.get_list_value('Misc', 'gpu_list'),
      epoch_step=cfg.get_int_value('Misc', 'epoch_step'),  # record where we were within an epoch.
      steps_per_stats=cfg.get_int_value('Misc', 'steps_per_stats'),
      steps_per_external_eval=cfg.get_int_value('Misc', 'steps_per_external_eval'),
      share_vocab=cfg.get_boolean_value('Misc', 'share_vocab'),
      metrics=cfg.get_list_value('Misc', 'metrics'),
      log_device_placement=cfg.get_boolean_value('Misc', 'log_device_placement'),
      random_seed=cfg.get_int_value('Misc', 'random_seed'),
  )

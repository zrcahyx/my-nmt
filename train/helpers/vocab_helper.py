#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility to handle vocabularies."""
import tensorflow as tf


UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def create_vocab2idx_tables(src_vocab_file, tgt_vocab_file, share_vocab):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  src_vocab2idx_table = tf.contrib.lookup.index_table_from_file(
      src_vocab_file, default_value=UNK_ID)
  if share_vocab:
    tgt_vocab2idx_table = src_vocab2idx_table
  else:
    tgt_vocab2idx_table = tf.contrib.lookup.index_table_from_file(
        tgt_vocab_file, default_value=UNK_ID)
  return src_vocab2idx_table, tgt_vocab2idx_table


def create_idx2vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  src_idx2vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
      src_vocab_file, default_value=UNK)
  if share_vocab:
    tgt_idx2vocab_table = src_idx2vocab_table
  else:
    tgt_idx2vocab_table = tf.contrib.lookup.index_to_string_table_from_file(
        tgt_vocab_file, default_value=UNK)
  return src_idx2vocab_table, tgt_idx2vocab_table


def vocab_test():
    pass


if __name__ == '__main__':
    vocab_test()

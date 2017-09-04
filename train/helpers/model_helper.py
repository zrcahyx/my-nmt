#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for building models."""

import tensorflow as tf


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.contrib.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.contrib.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def create_emb_for_encoder_and_decoder(hparams):
    """Create word embeddings for encoder and decoder."""
    if hparams.share_vocab:
        if hparams.src_vocab_size != hparams.tgt_vocab_size:
            raise ValueError('share embedding but has different src/tgt vocab')
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable(
                'embedding_share',
                [hparams.src_vocab_size, hparams.src_embed_size])
            embedding_encoder, embedding_decoder = embedding, embedding
    else:
        with tf.variable_scope('encoder'):
            embedding_encoder = tf.get_variable(
                "embedding_encoder",
                [hparams.src_vocab_size, hparams.src_embed_size])

        with tf.variable_scope('decoder'):
            embedding_decoder = tf.get_variable(
                "embedding_decoder",
                [hparams.tgt_vocab_size, hparams.tgt_embed_size])
    return embedding_encoder, embedding_decoder


def _create_single_cell(hparams, mode, residual_connection=False, device_str=None):
    """Create an instance of a single RNN cell."""
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        dropout = hparams.dropout
    else:
        dropout = 0.0

    # Cell Type
    if hparams.unit_type in ['lstm', 'Lstm', 'LSTM']:
        single_cell = tf.contrib.rnn.LSTMCell(
            hparams.num_units,
            use_peepholes=hparams.use_peepholes,
            forget_bias=hparams.forget_bias)
    elif hparams.unit_type in ['gru', 'Gru', 'GRU']:
        single_cell = tf.contrib.rnn.GRUCell(hparams.num_units)
    else:
        raise ValueError('Unknown unit type {}!'.format(hparams.unit_type))

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))
    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
    # Device Wrapper
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
    return single_cell


def _create_cell_list(hparams, mode):
  """Create a list of RNN cells."""
  # Multi-GPU
  cell_list = []
  for i in range(hparams.num_layers):
    # last layers use residual connection
    if i >= hparams.num_layers - hparams.num_residual_layers:
        residual_connection = True
    else:
        residual_connection = False
    num_gpus = len(hparams.gpu_list)
    device_str = '/gpu:{}'.format(hparams.gpu_list[i % num_gpus])

    single_cell = _create_single_cell(
        hparams, mode=mode,
        residual_connection=residual_connection,
        device_str=device_str)
    cell_list.append(single_cell)
  return cell_list


def create_rnn_cell(hparams, mode):
    """Create multi-layer RNN cell."""
    cell_list = _create_cell_list(hparams, mode)
    # Single layer.
    if len(cell_list) == 1:
        return cell_list[0]
    # Multi layers
    else:
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def _build_bidirectional_rnn(hparams, mode, encoder_emb_inp, sequence_length):
    """Create biddirectional RNN cells."""
    tmp_hparams = hparams
    if hparams.num_layers % 2 != 0:
        raise ValueError('use bi encoder_type, but num_layers not even!')
    # divided by 2 to get same size encoder states to be used by decoder!
    tmp_hparams.num_layers = int(hparams.num_layers / 2)
    tmp_hparams.num_residual_layers = int(hparams.num_residual_layers / 2)

    fw_cells = create_rnn_cell(tmp_hparams, mode)
    bw_cells = create_rnn_cell(tmp_hparams, mode)
    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cells, bw_cells, encoder_emb_inp,
        sequence_length=sequence_length,
        time_major=tmp_hparams.time_major)
    return tf.concat(bi_outputs, -1), bi_state


def run_bidirectional_rnn(hparams, mode, encoder_emb_inp, sequence_length):
    """get multi-layer Bidirectional RNN outputs and states."""
    encoder_outputs, bi_encoder_state = (
        _build_bidirectional_rnn(hparams, mode, encoder_emb_inp, sequence_length))
    num_bi_layers = int(hparams.num_layers / 2)
    if num_bi_layers == 1:
        encoder_state = bi_encoder_state
    else:
        # alternatively concat forward and backward states
        # bi_encoder_state: nested tuple ((fws), (bws))
        encoder_state = []
        for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)
    return encoder_outputs, encoder_state


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
    """Create attention mechanism based on the attention_option."""
    # Mechanism
    if attention_option in ['luong', 'Luong', 'LUONG']:
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option in ['scaled_luong']:
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option in ['bahdanau', 'Bahdanau']:
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option in ['normed_bahdanau']:
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)
    return attention_mechanism


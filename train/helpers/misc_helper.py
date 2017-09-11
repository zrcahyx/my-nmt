#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generally useful utility functions."""
import collections
import tensorflow as tf
from vocab_helper import SOS, EOS


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
        use_peepholes=cfg.get_boolean_value('Networks', 'use_peepholes'),
        encoder_type=cfg.get_string_value('Networks', 'encoder_type'),
        num_residual_layers=cfg.get_int_value('Networks', 'num_residual_layers'),
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
            if cfg.get_string_value('Vocab', 'sos') else SOS),
        eos=(cfg.get_string_value('Vocab', 'eos')
            if cfg.get_string_value('Vocab', 'sos') else EOS),
        bpe_delimiter=cfg.get_string_value('Vocab', 'bpe_delimiter'),
        src_vocab_size=cfg.get_int_value('Vocab', 'src_vocab_size'),
        tgt_vocab_size=cfg.get_int_value('Vocab', 'tgt_vocab_size'),
        src_embed_size=cfg.get_int_value('Vocab', 'src_embed_size'),
        tgt_embed_size=cfg.get_int_value('Vocab', 'tgt_embed_size'),


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
        maximum_iterations=cfg.get_int_value('Misc', 'maximum_iterations'),
        save_path=cfg.get_string_value('Misc', 'save_path'),


        # Serving
        export_path=cfg.get_string_value('Serving', 'export_path'),
        model_name=cfg.get_string_value('Serving', 'model_name'),
        host=cfg.get_string_value('Serving', 'host'),
        port=cfg.get_string_value('Serving', 'port'),
        requested_threshold=cfg.get_string_value('Serving', 'requested_threshold'),
        request_timeout=cfg.get_string_value('Serving', 'request_timeout'),
    )


def test():
    tgt_bpe_str = 'Les frais de bagages à main de Fron@@ tier ne seront pas appliqués avant l&apos; été , bien qu&apos; aucune date n&apos; ait été fixée .'
    symbols = tgt_bpe_str.split()
    print(format_bpe_text(symbols))


if __name__ == '__main__':
    test()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from helpers.model_helper import *


class NMTModel(object):
    def __init__(self,
                 hparams,
                 mode,
                 inputs,
                 source_vocab_table,
                 target_vocab_table,
                 reverse_target_vocab_table=None):

        # initialize
        self.hparams = hparams
        self.mode = mode
        self.inputs = inputs
        self.source_vocab_table = source_vocab_table
        self.target_vocab_table = target_vocab_table
        self.reverse_target_vocab_table = reverse_target_vocab_table

        # Initializer
        initializer = get_initializer(
            hparams.init_op, hparams.random_seed, hparams.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        """Create the model."""
        # create embedings for encoder and decoder
        self.embedding_encoder, self.embedding_decoder = (
            create_emb_for_encoder_and_decoder(hparams))

        # create encoder cell and encoder_emb_inp and get encoder_state/outputs
        source = inputs.source
        if hparams.time_major:
            source = tf.transpose(inputs.source)
        self.encoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_encoder, source, name='encoder_emb_inp')

        if hparams.encoder_type in ['bi', 'Bi', 'BI']:
            self.encoder_outputs, self.encoder_state = run_bidirectional_rnn(
                hparams, mode, self.encoder_emb_inp,
                inputs.source_sequence_length
            )
        else:
            with tf.variable_scope('encoder_cell'):
                self.encoder_cell = create_rnn_cell(hparams, mode)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                self.encoder_cell, self.encoder_emb_inp,
                sequence_length=inputs.source_sequence_length,
                dtype = tf.float32, time_major=hparams.time_major
            )

        # create decoder cell
        self.decoder_cell, self.decoder_initial_state = (
            self._build_decoder_cell(self.encoder_outputs, self.encoder_state))
        # create projection layer
        with tf.variable_scope("decoder/output_projection"):
            self.projection_layer = layers_core.Dense(
                hparams.tgt_vocab_size, use_bias=False, name='output_projection')

        if mode != tf.contrib.learn.ModeKeys.INFER:
            # create decoder_emb_inp
            target_input = inputs.target_input
            if hparams.time_major:
                target_input = tf.transpose(target_input)
            self.decoder_emb_inp = tf.nn.embedding_lookup(
                self.embedding_decoder, target_input, name='decoder_emb_inp')
            # Helper
            self.helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_emb_inp,
                inputs.target_sequence_length, time_major=True)
            # Decoder
            self.decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder_cell, self.helper,
                self.decoder_initial_state,
                output_layer=self.projection_layer)
            # Dynamic decoding
            self.outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder,
                output_time_major=hparams.time_major,
                swap_memory=True,
                maximum_iterations=hparams.maximum_iterations
                )
            # [batch_size, time, output_size] / [time, batch_size, output_size]
            with tf.name_scope('logits'):
                self.logits = self.outputs.rnn_output
            with tf.name_scope('sample_ids'):
                self.sample_ids = self.outputs.sample_id
            with tf.name_scope('loss'):
                self.loss = self._compute_loss(self.logits)
            tf.summary.scalar('loss', self.loss)
            if mode == tf.contrib.learn.ModeKeys.TRAIN:
                # Calculate and clip gradients
                self.params = tf.trainable_variables()
                self.gradients = tf.gradients(self.loss, self.params)
                self.clipped_gradients, _ = tf.clip_by_global_norm(
                    self.gradients, hparams.max_gradient_norm)
                # Optimization
                self.optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
                self.train_op = self.optimizer.apply_gradients(
                    zip(self.clipped_gradients, self.params))
        # inference
        elif mode == tf.contrib.learn.ModeKeys.INFER:
            self.tgt_sos_id = tf.cast(
                target_vocab_table.lookup(tf.constant(hparams.sos)),
                tf.int32)
            self.tgt_eos_id = tf.cast(
                target_vocab_table.lookup(tf.constant(hparams.eos)),
                tf.int32)
            self.start_tokens = tf.fill([self.hparams.batch_size], self.tgt_sos_id)
            self.end_token = self.tgt_eos_id
            self.decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=self.embedding_decoder,
                start_tokens=self.start_tokens,
                end_token=self.end_token,
                initial_state=self.decoder_initial_state,
                beam_width=hparams.beam_width,
                output_layer=self.projection_layer,
                length_penalty_weight=hparams.length_penalty_weight
            )
            # final_context_state: The final state of decoder RNN.
            self.outputs, self.final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=self.decoder,
                output_time_major=hparams.time_major,
                swap_memory=True,
                maximum_iterations=hparams.maximum_iterations
                )
            # [batch_size, beam_width, time]
            self.infer_ids = self.outputs.predicted_ids
            if hparams.time_major:
                # [time, batch_size, beam_width] -> [batch_size, beam_width, time]
                self.infer_ids = tf.transpose(self.infer_ids, perm=[1, 2, 0])


    def _build_decoder_cell(self, encoder_outputs, encoder_state):
        """Build an RNN cell with attention that can be used by decoder."""
        cell = create_rnn_cell(self.hparams, self.mode)
        attention_option = self.hparams.attention
        attention_architecture = self.hparams.attention_architecture

        if attention_architecture != "standard":
            raise ValueError(
                "Unknown attention architecture %s" % attention_architecture)

        if self.hparams.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        beam_width = self.hparams.beam_width
        source_sequence_length = self.inputs.source_sequence_length
        if (self.mode == tf.contrib.learn.ModeKeys.INFER and
            self.hparams.beam_width > 0):
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(
                source_sequence_length, multiplier=beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=beam_width)
            batch_size = self.hparams.batch_size * beam_width
        else:
            batch_size = self.hparams.batch_size

        attention_mechanism = create_attention_mechanism(
            attention_option, self.hparams.num_units,
            memory, source_sequence_length)
        alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         beam_width == 0)
        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=self.hparams.num_units,
            alignment_history=alignment_history,
            name="attention")
        # whether pass encoder states to decoder
        if self.hparams.pass_hidden_state:
            decoder_initial_state = cell.zero_state(batch_size,
                dtype=tf.float32).clone(cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        return cell, decoder_initial_state


    def _get_max_time(self, tensor):
        # get time_axis shape
        time_axis = 0 if self.hparams.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]


    def _compute_loss(self, logits):
        """Compute optimization loss."""
        # [batch_size, time]
        target_output = self.inputs.target_output
        if self.hparams.time_major:
            target_output = tf.transpose(target_output)
        # time dimension
        max_time = self._get_max_time(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        # target_sequence_length: [batch_size]
        # target_weights: [batch_size, time]
        target_weights = tf.sequence_mask(
            self.inputs.target_sequence_length, max_time, dtype=logits.dtype)
        if self.hparams.time_major:
            target_weights = tf.transpose(target_weights)
        # crossent: [time/batch_size, batch_size/time]
        loss = tf.reduce_sum(
            crossent * target_weights) / tf.to_float(self.hparams.batch_size)
        return loss







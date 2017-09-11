#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import sys
from os.path import abspath, dirname, join
from misc_helper import format_bpe_text, format_text
import tensorflow as tf

sys.path.append(dirname(dirname(dirname(abspath(__file__)))))
from scripts import bleu, rouge


def get_translation(nmt_outputs, sent_id, tgt_eos, bpe_delimiter):
    """Given batch decoding outputs, select a sentence and turn to text."""
    # Select a sentence
    output = nmt_outputs[sent_id, :].tolist()
    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
       output = output[:output.index(tgt_eos)]
    if not bpe_delimiter:
        translation = format_text(output)
    else:  # BPE
        translation = format_bpe_text(output, delimiter=bpe_delimiter)
    return translation


def compute_bleu(sample_sentences, tgt_sentences, hparams):
    """Compute bleu score according to predicted and label ids."""
    # sample_sentences: [batch_size, time] / [time, batch_size]
    # tgt_sentences: [batch_size, time]
    if hparams.time_major:
        # [batch_size, time]
        sample_sentences = sample_sentences.T
    translation_corpus, reference_corpus = [], []
    for i in xrange(hparams.batch_size):
        sample_sentence = get_translation(sample_sentences, i, hparams.eos, hparams.bpe_delimiter)
        tgt_sentence = get_translation(tgt_sentences, i, hparams.eos, hparams.bpe_delimiter)
        translation_corpus.append(sample_sentence)
        reference_corpus.append([tgt_sentence])
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      reference_corpus, translation_corpus, max_order=4, smooth=False)
    return bleu_score

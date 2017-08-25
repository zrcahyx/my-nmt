#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os.path import abspath, dirname, exists, join
sys.path.append(dirname(dirname(abspath(__file__))))
from utils.config_helper import ConfigHelper


def write_header(fw):
    fw.write('unk\n')
    fw.write('<s>\n')
    fw.write('</s>\n')


def main():
    data_dir = join(dirname(dirname(abspath(__file__))), 'train', 'data', 'no_bpe')
    # generate src/tgt tables
    src_vocab_set, tgt_vocab_set = set(), set()
    with open(join(data_dir, 'train.en'), 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                src_vocab_set.add(word)
    with open(join(data_dir, 'train.fr'), 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                tgt_vocab_set.add(word)

    # write to config
    cfg = ConfigHelper()
    cfg.set_value('Data', 'no_bpe_src_vocab_size', len(src_vocab_set) + 3)
    cfg.set_value('Data', 'no_bpe_tgt_vocab_size', len(tgt_vocab_set) + 3)
    print('no_bpe_src_vocab_size: {}'.format(len(src_vocab_set) + 3))
    print('no_bpe_tgt_vocab_size: {}'.format(len(tgt_vocab_set) + 3))

    # write to src/tgt vocab file
    fw = open(join(data_dir, 'vocab.en'), 'w')
    write_header(fw)
    for vocab in list(src_vocab_set):
        fw.write('{}\n'.format(vocab))
    fw.close()
    fw = open(join(data_dir, 'vocab.fr'), 'w')
    write_header(fw)
    for vocab in list(tgt_vocab_set):
        fw.write('{}\n'.format(vocab))
    fw.close()
    print('generate no bpe vocab files successfully!')


if __name__ == '__main__':
    main()

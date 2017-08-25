#!/usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

OUTPUT_DIR="${1:-wmt_fr_en}"
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"
mkdir -p $OUTPUT_DIR_DATA

if [ ! -f "${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz" ]; then
  echo "Downloading Europarl v7. This may take a while..."
  curl -o ${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz \
    http://www.statmt.org/europarl/v7/fr-en.tgz
else
  echo "${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz already exists!"
fi

if [ ! -f "${OUTPUT_DIR_DATA}/common-crawl.tgz" ]; then
  echo "Downloading Common Crawl corpus. This may take a while..."
  curl -o ${OUTPUT_DIR_DATA}/common-crawl.tgz \
    http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
else
  echo "${OUTPUT_DIR_DATA}/common-crawl.tgz already exists!"
fi

if [ ! -f "${OUTPUT_DIR_DATA}/test.tgz" ]; then
  echo "Downloading dev/test sets"
  curl -o ${OUTPUT_DIR_DATA}/dev.tgz \
    http://www.statmt.org/wmt15/dev-v1.tgz
  curl -o ${OUTPUT_DIR_DATA}/test.tgz \
    http://www.statmt.org/wmt14/test-full.tgz
else
  echo "${OUTPUT_DIR_DATA}/dev.tgz and ${OUTPUT_DIR_DATA}/test.tgz already exit!"
fi

# Extract everything
if [ ! -d "${OUTPUT_DIR_DATA}/test" ]; then
  echo "Extracting all files..."
  mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-fr-en"
  tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-fr-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-fr-en"
  mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
  tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
  mkdir -p "${OUTPUT_DIR_DATA}/dev"
  tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
  mkdir -p "${OUTPUT_DIR_DATA}/test"
  tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"
else
  echo "All files are already extracted!"
fi

# Concatenate Training data
if [ ! -f "${OUTPUT_DIR}/train.en" ]; then
  cat "${OUTPUT_DIR_DATA}/europarl-v7-fr-en/europarl-v7.fr-en.en" \
    "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.fr-en.en" \
    > "${OUTPUT_DIR}/train.en"
  wc -l "${OUTPUT_DIR}/train.en"
else
  echo "${OUTPUT_DIR}/train.en already exits!"
fi

if [ ! -f "${OUTPUT_DIR}/train.fr" ]; then
  cat "${OUTPUT_DIR_DATA}/europarl-v7-fr-en/europarl-v7.fr-en.fr" \
    "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.fr-en.fr" \
    > "${OUTPUT_DIR}/train.fr"
  wc -l "${OUTPUT_DIR}/train.fr"
else
  echo "${OUTPUT_DIR}/train.fr already exits!"
fi

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Convert SGM files
# Convert newstest2014 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-fren-src.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/dev.en
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2014-fren-ref.fr.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/dev.fr

# Convert newstest2014 data into raw text format
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test-full/newstest2014-fren-src.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/test-full/test.en
${OUTPUT_DIR}/mosesdecoder/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test-full/newstest2014-fren-ref.fr.sgm \
  > ${OUTPUT_DIR_DATA}/test/test-full/test.fr

# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/dev/dev/dev.fr ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/dev/dev/dev.en ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test-full/test.fr ${OUTPUT_DIR}
cp ${OUTPUT_DIR_DATA}/test/test-full/test.en ${OUTPUT_DIR}

# Tokenize data - 分词
for f in ${OUTPUT_DIR}/*.fr; do
    echo "Tokenizing $f..."
    ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l fr -threads 8 < $f > ${f%.*}.tok.fr
done

for f in ${OUTPUT_DIR}/*.en; do
    echo "Tokenizing $f..."
    ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean all corpora - Finally we clean, limiting sentence length to 80
for f in ${OUTPUT_DIR}/*.en; do
    fbase=${f%.*}
    echo "Cleaning ${fbase}..."
    ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase fr en "${fbase}.clean" 1 80
done

# Generate Subword Units (BPE) - byte-pair encoding／BPE
# 将单词分隔成子词（sub-word）。BPE 迭代合并出现频率高的符号对（symbol pair），
# 最后将出现频率高的 n 元合并成一个单独的符号，进而有效去除非词表词（out-of-vocabulary-word）。
# 该技术最初用来处理罕见单词，但是子词单元的模型性能全面超过全词系统，
# 32000 个子词单元是最高效的单词数量（Denkowski & Neubig, 2017）。
# Clone Subword NMT
if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
fi

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.fr" "${OUTPUT_DIR}/train.tok.clean.en" | \
    ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en fr; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  echo -e "<unk>\n<s>\n</s>" > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"
  cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.en" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.fr" | \
    ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' >> "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

done

# Duplicate vocab file with language suffix
cp "${OUTPUT_DIR}/vocab.bpe.32000" "${OUTPUT_DIR}/vocab.bpe.32000.en"
cp "${OUTPUT_DIR}/vocab.bpe.32000" "${OUTPUT_DIR}/vocab.bpe.32000.fr"

echo "All done."

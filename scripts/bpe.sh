set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

OUTPUT_DIR="${1:-wmt_fr_en}"
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

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

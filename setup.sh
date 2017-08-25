# data prepare

## no bpe train data and vocab
if [ ! -f "scripts/wmt_fr_en/train.tok.clean.en" ]; then
    ./scripts/wmt_en_fr.sh
else
    echo 'no bpe already done'
fi
if [ ! -f "train/data/no_bpe/test.fr" ]; then
    mkdir -p train/data/no_bpe
    cp scripts/wmt_fr_en/train.tok.clean.en train/data/no_bpe/train.en
    cp scripts/wmt_fr_en/train.tok.clean.fr train/data/no_bpe/train.fr
    cp scripts/wmt_fr_en/dev.tok.clean.en train/data/no_bpe/dev.en
    cp scripts/wmt_fr_en/dev.tok.clean.fr train/data/no_bpe/dev.fr
    cp scripts/wmt_fr_en/test.tok.clean.en train/data/no_bpe/test.en
    cp scripts/wmt_fr_en/test.tok.clean.fr train/data/no_bpe/test.fr
else
    echo 'no bpe data already exists!'
fi
if [ ! -f "train/data/no_bpe/vocab.fr" ]; then
    python scripts/generate_no_bpe_vocab_file.py
else
    echo 'no bpe vocab file already exists!'
fi

## bpe train data and voca
if [ ! -f "scripts/wmt_fr_en/train.tok.clean.bpe.32000.fr" ]; then
    ./scripts/bpe.sh
else
    echo 'bpe already done!'
fi
if [ ! -f "train/data/bpe/vocab.bpe.32000.fr" ]; then
    mkdir -p train/data/bpe
    cp scripts/wmt_fr_en/train.tok.clean.bpe.32000.fr train/data/bpe/train.fr
    cp scripts/wmt_fr_en/train.tok.clean.bpe.32000.en train/data/bpe/train.en
    cp scripts/wmt_fr_en/dev.tok.clean.bpe.32000.fr train/data/bpe/dev.fr
    cp scripts/wmt_fr_en/dev.tok.clean.bpe.32000.en train/data/bpe/dev.en
    cp scripts/wmt_fr_en/test.tok.clean.bpe.32000.fr train/data/bpe/test.fr
    cp scripts/wmt_fr_en/test.tok.clean.bpe.32000.en train/data/bpe/test.en
    cp scripts/wmt_fr_en/bpe.32000 train/data/bpe/
    cp scripts/wmt_fr_en/vocab.bpe.32000 train/data/bpe/
    cp scripts/wmt_fr_en/vocab.bpe.32000.en train/data/bpe/
    cp scripts/wmt_fr_en/vocab.bpe.32000.fr train/data/bpe/
else
    echo 'bpe data already exists!'
fi

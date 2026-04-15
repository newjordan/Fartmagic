#!/usr/bin/env bash
# 12L eval fix — identical conditions to midnight_12l_clean, only the eval pipeline changes.
set -euo pipefail

export COMPRESSOR=brotli
export NUM_LAYERS=12
export QUANT_ATTN_BITS=5
export QUANT_MLP_BITS=6
export QUANT_AUX_BITS=6
export QUANT_EMBED_BITS=8
export QUANT_OTHER_BITS=8
export LOADER_MODE=coprime
export COPRIME_MAX_LOADED_SHARDS=1
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64
export COMPLEMENT_ALPHA=0
export XSA_LAST_N=11
export BIGRAM_VOCAB_SIZE=2048
export ROPE_DIMS=16
export SWA_EVERY=50
export MTP_NUM_HEADS=0
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export CUBRIC_CADENCE=0
export NGRAM_ENTROPY_SHIFT=0
export VOCAB_SIZE=1024
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VE_ENABLED=0

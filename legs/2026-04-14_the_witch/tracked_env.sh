#!/usr/bin/env bash
# The Witch — wide + shallow + deep-looped architecture
# 3 physical layers at dim=1024, looped 4× = 12 effective layers
# Same param budget as 12L (dim=512, 12 layers), 87% H100 utilization (vs 43%)
set -euo pipefail

# Architecture: wide and shallow
export NUM_LAYERS=3
export MODEL_DIM=1024
export NUM_HEADS=16
export NUM_KV_HEADS=8
export MLP_MULT=3

# Looping: 4 passes through all 3 layers = 12 effective layers
export NUM_LOOPS=3
export LOOP_START=0
export LOOP_END=2
export ENABLE_LOOPING_AT=0.0

# Tokenizer: SP1024 (same as 12L)
export VOCAB_SIZE=1024
export DATA_PATH=./data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model

# Quantization: same policy as 12L
export COMPRESSOR=brotli
export QUANT_ATTN_BITS=5
export QUANT_MLP_BITS=6
export QUANT_AUX_BITS=6
export QUANT_EMBED_BITS=8
export QUANT_OTHER_BITS=8

# Data loader
export LOADER_MODE=coprime
export COPRIME_MAX_LOADED_SHARDS=1
export COPRIME_SHARDS_PER_BATCH=1
export COPRIME_SHARD_HOLD_STEPS=64

# Features carried from 12L
export COMPLEMENT_ALPHA=0
export XSA_LAST_N=3
export BIGRAM_VOCAB_SIZE=2048
export ROPE_DIMS=16
export SWA_EVERY=50
export MTP_NUM_HEADS=0
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export CUBRIC_CADENCE=0
export NGRAM_ENTROPY_SHIFT=0
export VE_ENABLED=0

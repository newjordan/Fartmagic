#!/usr/bin/env bash
# Edit this file, not the shell command line, when changing an experiment.
# This leg is a silo. If the test condition changes, create a new leg.
set -euo pipefail

export COMPRESSOR=brotli
export NUM_LAYERS=11
export QUANT_ATTN_BITS=6
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
export BIGRAM_VOCAB_SIZE=0
export ROPE_DIMS=16
export SWA_EVERY=50
export MTP_NUM_HEADS=0
export TRIGRAM=0
export NGRAM_EVAL_ORDER=0
export CUBRIC_CADENCE=0
export NGRAM_ENTROPY_SHIFT=0
export COMPILE_ENABLED=0
export COMPILE_FULLGRAPH=0
export GPTQ_CAL_MODE=online_cpu
export GPTQ_ONLINE_EVERY=4
export GPTQ_ONLINE_MAX_ROWS=4096
export GPTQ_ONLINE_REQUIRE_LOOPING=1

# Midnight III.V submission profile:
# - SP8192 vocab regime
# - lean physical stack with retained recurrence
# - remove aux vocab taxes for byte headroom
export VOCAB_SIZE=8192
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VE_ENABLED=0
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35

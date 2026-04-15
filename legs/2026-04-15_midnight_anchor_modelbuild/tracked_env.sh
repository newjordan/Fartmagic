#!/usr/bin/env bash
# Edit this file, not the shell command line, when changing an experiment.
# This leg is a silo. If the anchor contract changes, create a new leg.
set -euo pipefail

export NUM_LAYERS=11
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

# Midnight_Anchor model-build profile:
# - SP8192 vocab regime
# - retained recurrence
# - float checkpoint only; eval/compression live in sibling legs
export VOCAB_SIZE=8192
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export VE_ENABLED=0
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35
export EXPORT_MODEL_PATH=./artifacts/midnight_anchor/final_model_anchor.pt
export SKIP_GPTQ=1
export SKIP_FINAL_EVAL=1
export POST_EMA_DIAGNOSTIC=1

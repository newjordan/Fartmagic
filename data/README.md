# Data

This repo does not version datasets, tokenizer artifacts, or model weights.

FineWeb shards and SentencePiece tokenizers are downloaded separately:

```bash
python3 -m pip install -e ".[data]"
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

That writes:

- `data/tokenizers/`
- `data/datasets/`

`scripts/Im_sorry_pod_setup.sh` is the pod bootstrap. It calls the same downloader. It does not fetch model checkpoints.

Override the dataset repo with `MATCHED_FINEWEB_REPO_ID` and `MATCHED_FINEWEB_REMOTE_ROOT_PREFIX` when the manifest lives somewhere else. Training scripts read `DATA_PATH` and `TOKENIZER_PATH`.

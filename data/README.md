# Data

This repo does not version datasets or tokenizer artifacts.

Bootstrap them with:
```bash
bash scripts/pod_setup.sh
```

That script uses `data/cached_challenge_fineweb.py` to materialize:
- `data/tokenizers/`
- `data/datasets/`

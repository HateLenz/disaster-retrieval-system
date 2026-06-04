# Pretrained Models

Store third-party pretrained model snapshots here for offline training and evaluation.

The CLIP visual backbone used by this project is loaded from the local directory below by default:

```text
pretrained/openai-clip-vit-base-patch32/
```

No runtime download from Hugging Face is required. Place the full model snapshot in that folder before running training, feature extraction, or retrieval evaluation.

At minimum, keep the standard Transformers snapshot files together, including:

- `config.json`
- `preprocessor_config.json`
- `pytorch_model.bin`
- tokenizer files such as `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`

When you move the project to another machine or server, copy the `pretrained/` directory along with the codebase.

# `linkllama/training` (Axolotl, cap-50)

Full walkthrough: **[Retraining guide](../../assets/docs/retraining_guide.md)**.

Run from **`linkllama/training/`** (repo root contains `pyproject.toml`).

| File | Role |
|------|------|
| [`cf_lora_cap50.yml`](cf_lora_cap50.yml) | LoRA SFT on `chembl36_balanced_cap50.jsonl` |

```bash
cd linkllama/training
accelerate launch -m axolotl.cli.train cf_lora_cap50.yml
python -m axolotl.cli.merge_lora cf_lora_cap50.yml --lora_model_dir ./outputs_cap50
```

Dataset can be found at: https://huggingface.co/datasets/THGLab/LinkLlama-cap50-train

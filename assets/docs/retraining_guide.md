# Retraining

## 1. Corpus (ChEMBL → JSONL)

With conda env and `pip install -e .`:

```bash
python linkllama/llm/fragmentize.py input.smi --output-csv frags.csv
python linkllama/llm/calculate_properties.py frags.csv props.csv
python linkllama/llm/sft_corpus.py props.csv -o corpus.jsonl
```

Large files live under `data/` (gitignored). The **cap-50** training file used in the shipped Axolotl config is:

`data/chembl_processed/chembl36_balanced_cap50.jsonl`

Build it with [`linkllama/llm/create_balanced_subset.py`](../../linkllama/llm/create_balanced_subset.py) then `sft_corpus.py` — see [`linkllama/llm/README.md`](../../linkllama/llm/README.md).

If you publish this JSONL as a **separate Hugging Face dataset** repository, point training configs at the downloaded path or your Hub dataset id after `snapshot_download`.

## 2. LoRA (Axolotl)

Separate env with [Axolotl](https://docs.axolotl.ai/), `accelerate`, and `bitsandbytes`. Export `HF_TOKEN` for gated Llama weights; never commit tokens.

```bash
cd linkllama/training
accelerate launch -m axolotl.cli.train cf_lora_cap50.yml
python -m axolotl.cli.merge_lora cf_lora_cap50.yml --lora_model_dir ./outputs_cap50
```

Config: [`cf_lora_cap50.yml`](../../linkllama/training/cf_lora_cap50.yml).

## 3. Inference

After merging (or downloading from Hub), set `sampling.model_path` in [`data/inference_config.yaml`](../../data/inference_config.yaml) (or your copy of it) to your Hub model id or local path. See [inference_guide.md](inference_guide.md).

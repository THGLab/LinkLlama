# `linkllama/llm`

**Guides:** [Inference](../../assets/docs/inference_guide.md) · [Retraining](../../assets/docs/retraining_guide.md)

| Script | Role |
|--------|------|
| `fragmentize.py` | SMILES/CSV → fragments + geometry |
| `calculate_properties.py` | Property columns for CSV |
| `sft_corpus.py` | CSV → JSONL for SFT |
| `create_balanced_subset.py` | Cap (and optional hybrid) linker-frequency balancing |
| `inference.py` | Generation from CSV/SDF + YAML config |
| `constants.py` | Prompt templates |

ChEMBL-scale flow: fragmentize → properties → `sft_corpus.py`; optional **balanced** CSV then JSONL. Reference SLURM: `data/chembl_processed/prompt_response_generation/regenerate_chembl_workflow.sh` (edit paths for your cluster).

### Inference

Prefer **`data/inference_config.yaml`** at the repo root (all major YAML knobs). Example:

```bash
python linkllama/llm/inference.py --config data/inference_config.yaml --csv data.csv
```

### CLIs

Each script supports `--help`. Typical one-off pipeline:

```bash
python linkllama/llm/fragmentize.py input.smi --output-csv frags.csv
python linkllama/llm/calculate_properties.py frags.csv props.csv
python linkllama/llm/sft_corpus.py props.csv -o corpus.jsonl
```

# Inference

Load the cap-50 checkpoint from the **Hugging Face Hub**: [`THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50`](https://huggingface.co/THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50) (same id you pass to `AutoModelForCausalLM.from_pretrained`). You can instead set `sampling.model_path` to a **local directory** in Hugging Face layout if you exported weights offline.

## Config YAML

Use [`data/inference_config.yaml`](../../data/inference_config.yaml) at the repository root: **tunable fields** under `prompt` (linker type, rotatable bonds / heavy atoms, H-bond counts, MW / logP / TPSA bands, reasonability) and **sampling** (`model_path`, `num_samples`, `max_length`). Use `null` where unconditional.

`sampling.temperature` and `sampling.top_p` default to **1.4** and **0.99** (documented in the YAML). They are not overridable from the CLI; change them in your config copy if you need different generation behavior.

Copy and edit locally:

```bash
cp data/inference_config.yaml my_run.yaml
```

Use `huggingface-cli login` or `HF_TOKEN` for gated Hub / Llama access when needed.

## Run

From the **repository root** (where `pyproject.toml` lives):

```bash
python linkllama/llm/inference.py --config my_run.yaml --csv your_input.csv
```

Optional: `--num_samples` (overrides `sampling.num_samples` in the YAML).

**Input:** CSV or SDF per `inference.py --help`. GPU recommended.

**Minimal example** (two rows, same fragment convention as benchmark ZINC CSVs): [`data/zinc_minimal.csv`](../../data/zinc_minimal.csv). Quick smoke test (fewer samples than the default in the YAML):

```bash
python linkllama/llm/inference.py \
  --config data/inference_config.yaml \
  --csv data/zinc_minimal.csv \
  --num_samples 2
```

See [`data/README.md`](../../data/README.md) for what lives under `data/`.

More module detail: [`linkllama/llm/README.md`](../../linkllama/llm/README.md).

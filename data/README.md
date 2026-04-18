# Bundled data

| File | Purpose |
| --- | --- |
| [`inference_config.yaml`](inference_config.yaml) | Default inference YAML (`prompt` + `sampling` + `output`); copy to edit. |
| [`zinc_minimal.csv`](zinc_minimal.csv) | Two-row ZINC-style fragment CSV (`fragments`, `distance_angstrom`, `angle_degrees`) for a quick run. |

From the **repository root** (next to `pyproject.toml`):

```bash
python linkllama/llm/inference.py \
  --config data/inference_config.yaml \
  --csv data/zinc_minimal.csv \
  --num_samples 2
```

Outputs follow `output.output_dir` / `output_suffix` in the YAML (default under `./inference_outputs/`). See **[Inference guide](../assets/docs/inference_guide.md)** for full options.

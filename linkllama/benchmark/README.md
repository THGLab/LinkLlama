# `linkllama/benchmark`

Evaluates DiffLinker, DeLinker, and LinkLlama outputs: **unified** property suite and optional **UniDock** docking.

| File | Role |
|------|------|
| `unified_benchmark.py` | Main metrics (validity, QED, SA, RMSD, reasonability, …) |
| `unidock_benchmark.py` | UniDock driver for protein–ligand sets |
| `source_*.py` | Load model outputs per method |
| `geometry_benchmark_base.py` | Shared geometry / energy helpers |

```python
from linkllama.benchmark import run_benchmark, run_all_benchmarks
# See source for `run_benchmark` / `run_all_benchmarks` arguments.
```


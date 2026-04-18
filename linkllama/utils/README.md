# `linkllama/utils`

Supporting code for fragmentation, 3D geometry, conformers, and property filters (used by `linkllama/llm`). See module docstrings for APIs.

| Module | Role |
|--------|------|
| `fragmentation.py` | MMPA-style fragmentation (RDKit) |
| `geometry.py` | Distances/angles between fragments |
| `conformer_generation.py` | ETKDG conformers |
| `properties.py` | PAINS, REOS, ring checks, etc. |
| `sdf_reader.py` | SDF → CSV helpers |

Ring-frequency data: `ring_systems/chembl*.csv`.

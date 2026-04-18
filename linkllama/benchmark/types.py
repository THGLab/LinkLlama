from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union


MolReprType = Literal["smiles", "sdf_path"]
MolRepr = Union[str, Path]


@dataclass(frozen=True)
class GeneratedSample:
    """One generated candidate for an instance."""

    name: str
    fragments_smi: str
    gt_smiles: Optional[str]
    sample_id: str
    mol_repr_type: MolReprType
    mol_repr: MolRepr
    # Optional: if we still have a distinct linker (LinkLlama)
    linker_smiles: Optional[str] = None


@dataclass(frozen=True)
class InstanceSamples:
    """All generated candidates for one instance (one fragment pair)."""

    name: str
    fragments_smi: str
    gt_smiles: Optional[str]
    expected_total: Optional[int]
    samples: list[GeneratedSample]


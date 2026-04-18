from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .types import GeneratedSample, InstanceSamples


def subfolder_to_name(subfolder_name: str) -> str:
    """fragments_0_1k_zinc_0 -> 1k_zinc_0"""
    if subfolder_name.startswith("fragments_0_"):
        return subfolder_name.replace("fragments_0_", "", 1)
    return subfolder_name


def load_instances(
    input_dir: Path,
    name_to_fragments: Dict[str, str],
    name_to_gt_smiles: Dict[str, str],
    max_samples_per_instance: Optional[int] = None,
) -> List[InstanceSamples]:
    """
    DiffLinker input: directory of subfolders (fragments_0_<name>) each with many .sdf files.
    Returns one InstanceSamples per subfolder.
    """
    input_dir = Path(input_dir)
    instances: List[InstanceSamples] = []
    for subdir in sorted(d for d in input_dir.iterdir() if d.is_dir()):
        name = subfolder_to_name(subdir.name)
        frags = name_to_fragments.get(name)
        if not frags:
            continue
        gt = name_to_gt_smiles.get(name)
        sdf_files = sorted(subdir.glob("*.sdf"))
        if max_samples_per_instance is not None:
            sdf_files = sdf_files[: max_samples_per_instance]
        samples = [
            GeneratedSample(
                name=name,
                fragments_smi=frags,
                gt_smiles=gt,
                sample_id=sdf_path.stem,
                mol_repr_type="sdf_path",
                mol_repr=sdf_path,
            )
            for sdf_path in sdf_files
        ]
        instances.append(
            InstanceSamples(
                name=name,
                fragments_smi=frags,
                gt_smiles=gt,
                expected_total=None,
                samples=samples,
            )
        )
    return instances


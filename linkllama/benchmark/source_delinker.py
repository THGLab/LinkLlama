from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .types import GeneratedSample, InstanceSamples


def parse_delinker_smi(smi_path: Path) -> List[Tuple[str, str, str]]:
    """Each line: fragments gt generated (space-separated)."""
    triples: List[Tuple[str, str, str]] = []
    with open(smi_path) as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 3:
                continue
            triples.append((toks[0], toks[1], toks[2]))
    return triples


def load_instances(
    input_path: Path,
    name_to_fragments: Dict[str, str],
    name_to_gt_smiles: Dict[str, str],
    expected_samples_per_instance: Optional[int] = 100,
    max_samples_per_instance: Optional[int] = None,
) -> List[InstanceSamples]:
    """
    DeLinker input:
      - directory of subfolders named by instance (each has generated.smi), OR
      - a single .smi file with triples.

    For subdir mode, expected_total is used as the denominator (like benchmark_analysis_delinker.py).
    """
    input_path = Path(input_path)
    instances: List[InstanceSamples] = []

    if input_path.is_dir():
        for subdir in sorted(d for d in input_path.iterdir() if d.is_dir()):
            name = subdir.name
            frags = name_to_fragments.get(name)
            if not frags:
                continue
            gt = name_to_gt_smiles.get(name)
            smi_path = subdir / "generated.smi"
            if not smi_path.exists():
                samples: List[GeneratedSample] = []
            else:
                triples = parse_delinker_smi(smi_path)
                if max_samples_per_instance is not None:
                    triples = triples[: max_samples_per_instance]
                samples = [
                    GeneratedSample(
                        name=name,
                        fragments_smi=frags,
                        gt_smiles=gt,
                        sample_id=f"line_{i}",
                        mol_repr_type="smiles",
                        mol_repr=gen,
                    )
                    for i, (_fr, _gt, gen) in enumerate(triples)
                ]
            instances.append(
                InstanceSamples(
                    name=name,
                    fragments_smi=frags,
                    gt_smiles=gt,
                    expected_total=expected_samples_per_instance,
                    samples=samples,
                )
            )
        return instances

    # Single .smi file: group by fragments, map to name via CSV fragments mapping.
    frag_to_name = {v: k for k, v in name_to_fragments.items()}
    frag_to_gt = {v: name_to_gt_smiles.get(k) for k, v in name_to_fragments.items()}
    grouped: Dict[str, List[str]] = {}
    for frags, _gt, gen in parse_delinker_smi(input_path):
        grouped.setdefault(frags, []).append(gen)

    for frags, gens in grouped.items():
        name = frag_to_name.get(frags)
        if name is None:
            continue
        gt = frag_to_gt.get(frags)
        if max_samples_per_instance is not None:
            gens = gens[: max_samples_per_instance]
        samples = [
            GeneratedSample(
                name=name,
                fragments_smi=frags,
                gt_smiles=gt,
                sample_id=f"line_{i}",
                mol_repr_type="smiles",
                mol_repr=gen,
            )
            for i, gen in enumerate(gens)
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


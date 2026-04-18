from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem

from linkllama.utils.fragmentation import join_fragments_linker

from .types import GeneratedSample, InstanceSamples


def load_instances(
    pkl_path: Path,
    name_to_fragments: Dict[str, str],
    name_to_gt_smiles: Dict[str, str],
    run_index: Optional[int] = None,
    max_samples_per_instance: Optional[int] = None,
    failed_cases_path: Optional[Path] = None,
) -> List[InstanceSamples]:
    """
    LinkLlama input: pkl with {key: {data: {fragments}, responses: [{linker, ...}, ...]}}.

    If run_index is None: include ALL responses per instance.
    If run_index is set: include only that response index (if present).
    If failed_cases_path is set: write a summary of skipped/failed cases (one per line).
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    frag_to_name = {v: k for k, v in name_to_fragments.items()}
    instances: List[InstanceSamples] = []
    failed_cases: List[tuple[str, str]] = []  # (identifier, reason)

    for key, result in results.items():
        if isinstance(result, dict) and result.get("error"):
            failed_cases.append((str(key)[:120], "error"))
            continue
        row_data = result.get("data", {}) if isinstance(result, dict) else {}
        fragments_smi = row_data.get("fragments", "") if isinstance(row_data, dict) else ""
        if not fragments_smi and isinstance(key, str) and "." in key and "*" in key:
            fragments_smi = key
        if not fragments_smi:
            failed_cases.append((str(key)[:120], "no_fragments"))
            continue
        name = frag_to_name.get(fragments_smi)
        if name is None:
            failed_cases.append((str(key)[:120], "no_name_match"))
            continue
        gt = name_to_gt_smiles.get(name)
        responses = result.get("responses", []) if isinstance(result, dict) else []
        if run_index is not None:
            responses = [responses[run_index]] if 0 <= run_index < len(responses) else []
        if max_samples_per_instance is not None:
            responses = responses[: max_samples_per_instance]

        samples: List[GeneratedSample] = []
        for i, resp in enumerate(responses):
            if not isinstance(resp, dict) or resp.get("error"):
                continue
            linker = resp.get("linker")
            if not linker or not isinstance(linker, str):
                continue
            try:
                joined = join_fragments_linker(fragments_smi, linker.strip())
            except Exception:
                continue
            if not joined:
                continue
            mol = Chem.MolFromSmiles(joined)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            samples.append(
                GeneratedSample(
                    name=name,
                    fragments_smi=fragments_smi,
                    gt_smiles=gt,
                    sample_id=f"resp_{i}",
                    mol_repr_type="smiles",
                    mol_repr=joined,
                    linker_smiles=linker.strip(),
                )
            )

        if len(samples) == 0:
            failed_cases.append((name, "no_valid_responses"))

        instances.append(
            InstanceSamples(
                name=name,
                fragments_smi=fragments_smi,
                gt_smiles=gt,
                expected_total=None,
                samples=samples,
            )
        )

    if failed_cases_path is not None and failed_cases:
        failed_cases_path = Path(failed_cases_path)
        failed_cases_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# LinkLlama failed/skipped cases (identifier, reason)",
            "# reason: error | no_fragments | no_name_match | no_valid_responses",
            "",
        ] + [f"{ident}\t{reason}" for ident, reason in failed_cases]
        failed_cases_path.write_text("\n".join(lines), encoding="utf-8")

    return instances


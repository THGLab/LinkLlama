from __future__ import annotations

import math
import os
import pickle
import re
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED as rdQED
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

from rdkit.Chem import RDConfig

import sys

sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore

from linkllama.utils.properties import (
    has_undesirable_pattern,
    has_pains_alert,
    check_reos,
    has_bad_ring,
    check_ring_system,
)
from linkllama.utils.fragmentation import get_linker

from .geometry_benchmark_base import GeometryBenchmark, EnergyRMSDResult, build_clean_frag_charged

from .types import GeneratedSample, InstanceSamples


def canonicalize_molecule(smi: str) -> Optional[str]:
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def canonicalize_linker(smi: str) -> Optional[str]:
    """Canonicalize linker SMILES without stereochemistry (for novelty)."""
    if not smi or not isinstance(smi, str):
        return None
    smi = smi.strip()
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_to_compact_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    if mol is None:
        return None
    try:
        rw = Chem.RWMol()
        for a in mol.GetAtoms():
            new_atom = Chem.Atom(a.GetSymbol())
            new_atom.SetFormalCharge(a.GetFormalCharge())
            rw.AddAtom(new_atom)
        for b in mol.GetBonds():
            rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
        Chem.SanitizeMol(rw)
        Chem.RemoveStereochemistry(rw)
        return Chem.MolToSmiles(rw, canonical=True)
    except Exception:
        return None


def calculate_qed(mol: Chem.Mol) -> Optional[float]:
    try:
        return float(rdQED.qed(mol))
    except Exception:
        return None


def calculate_sa(mol: Chem.Mol) -> Optional[float]:
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def fragment_smiles_to_query_mol(frag_smi: str) -> Optional[Chem.Mol]:
    """Neutral query for property validity: dummies -> H, charges cleared."""
    if not frag_smi:
        return None
    try:
        q = re.sub(r"\[\*:[12]\]", "[H]", frag_smi)
        mol = Chem.MolFromSmiles(q)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetFormalCharge(0)
        return mol
    except Exception:
        return None


def mol_contains_all_fragment_atoms(mol: Chem.Mol, fragments_smi: str) -> bool:
    if mol is None or not fragments_smi:
        return False
    try:
        parts = fragments_smi.split(".")
        if len(parts) != 2:
            return False
        q1 = fragment_smiles_to_query_mol(parts[0].strip())
        q2 = fragment_smiles_to_query_mol(parts[1].strip())
        if q1 is None or q2 is None:
            return False
        return mol.HasSubstructMatch(q1) and mol.HasSubstructMatch(q2)
    except Exception:
        return False


def compute_reasonability(mol: Chem.Mol, smi: str) -> Dict[str, Any]:
    mol_has_undes = has_undesirable_pattern(mol)
    mol_has_pains = has_pains_alert(mol)
    reos_rule = str(check_reos(smi)).strip()
    mol_bad_ring = has_bad_ring(mol)
    problematic_ring = str(check_ring_system(smi)).strip()
    pass_undes = not mol_has_undes
    pass_pains = not mol_has_pains
    pass_reos = reos_rule in ("", "nan")
    pass_bad = not mol_bad_ring
    pass_prob = problematic_ring in ("", "nan")
    pass_all = pass_undes and pass_pains and pass_reos and pass_bad and pass_prob
    return {
        "pass_undes": pass_undes,
        "pass_pains": pass_pains,
        "pass_reos": pass_reos,
        "pass_bad_ring": pass_bad,
        "pass_problematic_ring": pass_prob,
        "pass_all": pass_all,
    }


def _geometry_instance_worker(
    args: Tuple[str, List[Tuple[str, Any]], str, str]
) -> Optional[Tuple[str, List[float], List[float]]]:
    """
    Worker for one instance: (inst_name, samples_data, csv_path, reference_sdfs_dir).
    samples_data: list of (mol_repr_type, mol_repr) with mol_repr as str path or SMILES.
    Runs geometry on all samples; returns (inst_name, rmsd_list, energy_delta_list)
    with finite energy deltas only, or None if no valid geometry results.
    """
    inst_name, samples_data, csv_path_str, ref_sdfs_str = args
    bench = GeometryBenchmark(Path(csv_path_str), Path(ref_sdfs_str))
    rmsd_list: List[float] = []
    ed_list: List[float] = []
    for mol_repr_type, mol_repr in samples_data:
        if mol_repr_type == "sdf_path":
            supp = Chem.SDMolSupplier(str(mol_repr), removeHs=False)
            mol0 = next((m for m in supp if m), None)
            if mol0 is None:
                continue
            mol = Chem.RemoveHs(Chem.Mol(mol0))
            try:
                mol = Chem.AddHs(mol)  # prepare with full valence for geometry
            except Exception:
                continue
            res = bench.process_mol(mol, inst_name)
        else:
            res = bench.process_smiles(str(mol_repr), inst_name)
        if res is None:
            continue
        rmsd_list.append(res.rmsd)
        if not (isinstance(res.energy_delta, float) and math.isnan(res.energy_delta)):
            ed_list.append(res.energy_delta)
    if not rmsd_list:
        return None
    return (inst_name, rmsd_list, ed_list)


def _sample_worker(args: Tuple[GeneratedSample, int]) -> Tuple[int, Dict[str, Any]]:
    sample, pos = args
    out: Dict[str, Any] = {
        "name": sample.name,
        "sample_id": sample.sample_id,
        "valid": False,
        "canon_smiles": None,
        "canon_linker": None,
        "qed": None,
        "sa": None,
        "recovered": False,
        "pass_undes": False,
        "pass_pains": False,
        "pass_reos": False,
        "pass_bad_ring": False,
        "pass_problematic_ring": False,
        "pass_all": False,
        "frag_charge_match": False,
    }
    try:
        if sample.mol_repr_type == "smiles":
            smi = str(sample.mol_repr).strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                return pos, out
        else:
            sdf_path = Path(sample.mol_repr)
            supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            mol0 = next(iter(supp), None)
            if mol0 is None:
                return pos, out
            mol = Chem.RemoveHs(mol0)
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)  # prepare with full valence (DiffLinker SDFs)
            except Exception:
                return pos, out

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return pos, out

        # Validity = parseable by RDKit + contains fragments with correct charge state
        clean_frag = build_clean_frag_charged(sample.fragments_smi)
        if clean_frag is None or len(mol.GetSubstructMatches(clean_frag)) == 0:
            return pos, out

        out["valid"] = True
        out["frag_charge_match"] = True  # redundant with valid; kept for any reader
        canon = mol_to_compact_canonical_smiles(mol)
        out["canon_smiles"] = canon
        if sample.linker_smiles:
            out["canon_linker"] = canonicalize_linker(sample.linker_smiles)
        elif canon:
            try:
                mol_l = Chem.MolFromSmiles(canon)
                if mol_l is not None:
                    Chem.SanitizeMol(mol_l)
                    raw_ls = get_linker(mol_l, clean_frag, sample.fragments_smi)
                    if raw_ls:
                        out["canon_linker"] = canonicalize_linker(raw_ls)
            except Exception:
                pass
        out["qed"] = calculate_qed(mol)
        out["sa"] = calculate_sa(mol)
        out.update(compute_reasonability(mol, canon if canon else ""))

        gt_canon = canonicalize_molecule(sample.gt_smiles) if sample.gt_smiles else None
        out["recovered"] = bool(gt_canon and canon and gt_canon == canon)
        return pos, out
    except Exception:
        return pos, out


@dataclass
class UnifiedBenchmarkResult:
    method: str
    n_instances: int
    total_samples: int
    subfolders_failed: int
    n_expected_instances: int
    n_missing_instances: int
    n_instances_zero_validity: int
    n_samples_invalid: int
    mol_valid_pct_mean: float
    mol_valid_pct_std: float
    uniqueness_pct_mean: float
    uniqueness_pct_std: float
    recovery_total_pct: float
    mean_mol_qed: Optional[float]
    std_mol_qed: Optional[float]
    mean_mol_sa: Optional[float]
    std_mol_sa: Optional[float]
    novelty_pct_mean: Optional[float] = None
    novelty_pct_std: Optional[float] = None
    overall_success_rate_pct: Optional[float] = None  # total_valid / total_samples (sample-level)
    folder_success_rate_pct: Optional[float] = None  # n_success / n_expected_instances (instance-level)
    # Reasonability: component rates pooled over all valid mols; pass_all headline is
    # mean ± std of per-instance % pass_all (same instance cohort as QED/SA).
    n_reasonability: int = 0
    pct_pass_undesirable: float = 0.0
    pct_pass_pains: float = 0.0
    pct_pass_reos: float = 0.0
    pct_pass_bad_ring: float = 0.0
    pct_pass_problematic_ring: float = 0.0
    pct_pass_all_reasonability: float = 0.0
    std_pct_pass_all_reasonability: Optional[float] = None
    # Geometry (DeLinker-style fragment RMSD)
    n_geometry: int = 0
    mean_rmsd: Optional[float] = None
    std_rmsd: Optional[float] = None
    mean_best_rmsd: Optional[float] = None
    std_best_rmsd: Optional[float] = None
    mean_energy_delta: Optional[float] = None
    std_energy_delta: Optional[float] = None
    energy_delta_p25: Optional[float] = None
    energy_delta_p50: Optional[float] = None
    energy_delta_p75: Optional[float] = None
    per_instance: List[Dict[str, Any]] = field(default_factory=list)


def _load_training_linkers_pkl(pkl_path: Optional[Path]) -> Optional[Set[str]]:
    """Load set of canonical linker SMILES from pkl (for novelty)."""
    if not pkl_path or not Path(pkl_path).exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return set(data) if not isinstance(data, set) else data
    except Exception:
        return None


class UnifiedBenchmark:
    """
    Unified benchmark with one class that computes:
    validity, uniqueness, recovery, QED, SA, novelty, reasonability, RMSD, energy.

    - Validity = parseable by RDKit and contains fragments with correct charge state (charged
      scaffold match). Uniqueness, QED, SA, reasonability are computed over valid molecules only.
    - Recovery is computed over instances (any valid sample matching GT counts as recovered).
    - Geometry uses GeometryBenchmark on valid samples (same fragment+charge precondition).
    - Novelty: optional training_linkers_pkl; % of distinct canonical linkers not in training inventory.
    """

    def __init__(
        self,
        csv_path: Path,
        reference_sdfs_dir: Optional[Path] = None,
        training_linkers_pkl: Optional[Path] = None,
    ) -> None:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        self.name_to_fragments: Dict[str, str] = {}
        self.name_to_gt_smiles: Dict[str, str] = {}
        for _, row in df.iterrows():
            name = row.get("name")
            if pd.isna(name):
                continue
            name = str(name).strip()
            frags = row.get("fragments", "")
            if pd.notna(frags):
                self.name_to_fragments[name] = str(frags).strip()
            if "SMILES" in row and pd.notna(row.get("SMILES")):
                self.name_to_gt_smiles[name] = str(row["SMILES"]).strip()
        self._csv_path = csv_path
        self._geometry: Optional[GeometryBenchmark] = None
        self._reference_sdfs_dir: Optional[Path] = None
        if reference_sdfs_dir is not None:
            reference_sdfs_dir = Path(reference_sdfs_dir)
            if reference_sdfs_dir.exists():
                self._geometry = GeometryBenchmark(csv_path, reference_sdfs_dir)
                self._reference_sdfs_dir = reference_sdfs_dir
        self._training_linkers_pkl: Optional[Path] = (
            Path(training_linkers_pkl) if training_linkers_pkl else None
        )

    def run(
        self,
        method: str,
        instances: List[InstanceSamples],
        n_workers: int = 1,
        geometry_max_candidates: Optional[int] = None,
        compute_geometry: bool = True,
        expected_samples_per_instance: Optional[int] = None,
    ) -> UnifiedBenchmarkResult:
        # Flatten samples for worker pool
        flat: List[GeneratedSample] = []
        for inst in instances:
            flat.extend(inst.samples)
        # total_samples = n_instances * expected per instance (when provided)
        if expected_samples_per_instance is not None:
            total_samples = len(instances) * expected_samples_per_instance
        else:
            total_samples = len(flat)

        # Compute properties + reasonability per sample (parallelizable)
        per_sample_by_pos: List[Optional[Dict[str, Any]]] = [None] * len(flat)
        tasks = [(s, i) for i, s in enumerate(flat)]
        if n_workers > 1 and tasks:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_sample_worker, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Props+Reason"):
                    pos, res = fut.result()
                    per_sample_by_pos[pos] = res
        else:
            for t in tqdm(tasks, desc="Props+Reason"):
                pos, res = _sample_worker(t)
                per_sample_by_pos[pos] = res

        # (name, sample_id) -> result for geometry filtering
        sample_key_to_result: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for i, s in enumerate(flat):
            r = per_sample_by_pos[i]
            if r is not None:
                sample_key_to_result[(s.name, s.sample_id)] = r

        # Group sample results by instance name (order matches flat, so inst.samples order)
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for i in range(len(flat)):
            res = per_sample_by_pos[i]
            if res is not None:
                by_name.setdefault(flat[i].name, []).append(res)

        per_instance: List[Dict[str, Any]] = []
        success_instances: List[Dict[str, Any]] = []
        recovered_success = 0

        # Reasonability counts over all valid molecules (pooled component rates)
        n_reason = 0
        pass_undes = pass_pains = pass_reos = pass_bad = pass_prob = 0

        training_linkers: Optional[Set[str]] = None
        if self._training_linkers_pkl:
            training_linkers = _load_training_linkers_pkl(self._training_linkers_pkl)

        for inst in instances:
            results = by_name.get(inst.name, [])
            denom = inst.expected_total if inst.expected_total is not None else len(results)
            valid = [r for r in results if r.get("valid")]
            n_valid = len(valid)
            mol_valid_pct = 100.0 * n_valid / denom if denom and denom > 0 else 0.0
            canon_list = [r["canon_smiles"] for r in valid if r.get("canon_smiles")]
            uniqueness_pct = 100.0 * len(set(canon_list)) / len(canon_list) if canon_list else 0.0
            recovered = any(r.get("recovered") for r in valid)
            mean_qed = None
            mean_sa = None
            std_mol_qed: Optional[float] = None
            std_mol_sa: Optional[float] = None
            qeds = [r["qed"] for r in valid if r.get("qed") is not None]
            sas = [r["sa"] for r in valid if r.get("sa") is not None]
            if qeds:
                mean_qed = float(sum(qeds) / len(qeds))
                std_mol_qed = float(statistics.stdev(qeds)) if len(qeds) >= 2 else None
            if sas:
                mean_sa = float(sum(sas) / len(sas))
                std_mol_sa = float(statistics.stdev(sas)) if len(sas) >= 2 else None

            mean_pass_all_reasonability: Optional[float] = None
            pct_pass_undesirable_i: Optional[float] = None
            pct_pass_pains_i: Optional[float] = None
            pct_pass_reos_i: Optional[float] = None
            pct_pass_bad_ring_i: Optional[float] = None
            pct_pass_problematic_ring_i: Optional[float] = None
            if n_valid:
                mean_pass_all_reasonability = (
                    100.0 * sum(1 for r in valid if r.get("pass_all")) / n_valid
                )
                pct_pass_undesirable_i = 100.0 * sum(1 for r in valid if r.get("pass_undes")) / n_valid
                pct_pass_pains_i = 100.0 * sum(1 for r in valid if r.get("pass_pains")) / n_valid
                pct_pass_reos_i = 100.0 * sum(1 for r in valid if r.get("pass_reos")) / n_valid
                pct_pass_bad_ring_i = 100.0 * sum(1 for r in valid if r.get("pass_bad_ring")) / n_valid
                pct_pass_problematic_ring_i = (
                    100.0 * sum(1 for r in valid if r.get("pass_problematic_ring")) / n_valid
                )

            novelty_pct_i: Optional[float] = None
            if training_linkers is not None and n_valid > 0:
                inst_linkers = [r["canon_linker"] for r in valid if r.get("canon_linker")]
                if inst_linkers:
                    unique_linkers = set(inst_linkers)
                    n_train = sum(1 for c in unique_linkers if c in training_linkers)
                    novelty_pct_i = 100.0 * (len(unique_linkers) - n_train) / len(unique_linkers)

            for r in valid:
                n_reason += 1
                if r.get("pass_undes"):
                    pass_undes += 1
                if r.get("pass_pains"):
                    pass_pains += 1
                if r.get("pass_reos"):
                    pass_reos += 1
                if r.get("pass_bad_ring"):
                    pass_bad += 1
                if r.get("pass_problematic_ring"):
                    pass_prob += 1

            row = {
                "name": inst.name,
                "total_samples": denom if denom is not None else len(results),
                "n_valid": n_valid,
                "mol_valid_pct": mol_valid_pct,
                "uniqueness_pct": uniqueness_pct,
                "recovered": recovered,
                "mean_mol_qed": mean_qed,
                "std_mol_qed": std_mol_qed,
                "mean_mol_sa": mean_sa,
                "std_mol_sa": std_mol_sa,
                "mean_pass_all_reasonability": mean_pass_all_reasonability,
                "pct_pass_undesirable": pct_pass_undesirable_i,
                "pct_pass_pains": pct_pass_pains_i,
                "pct_pass_reos": pct_pass_reos_i,
                "pct_pass_bad_ring": pct_pass_bad_ring_i,
                "pct_pass_problematic_ring": pct_pass_problematic_ring_i,
                "novelty_pct": novelty_pct_i,
            }
            per_instance.append(row)
            if mol_valid_pct > 0:
                success_instances.append(row)
                if recovered:
                    recovered_success += 1

        n_instances = len(per_instance)
        n_success = len(success_instances)
        subfolders_failed = n_instances - n_success
        n_expected_instances = len(self.name_to_fragments)
        n_missing_instances = max(0, n_expected_instances - n_instances)
        n_instances_zero_validity = subfolders_failed
        total_valid_samples = sum(r["n_valid"] for r in per_instance)
        n_samples_invalid = total_samples - total_valid_samples
        # Recovery: denominator = total expected instances (e.g. 1000 for zinc)
        recovery_total_pct = (
            100.0 * recovered_success / n_expected_instances
            if n_expected_instances and n_expected_instances > 0
            else 0.0
        )
        overall_success_rate_pct = (
            100.0 * total_valid_samples / total_samples if total_samples > 0 else None
        )
        # Success rate: % of instances (folders) with ≥1 valid sample
        folder_success_rate_pct = (
            100.0 * n_success / n_expected_instances
            if n_expected_instances and n_expected_instances > 0
            else None
        )
        # Validity & uniqueness: only over instances with ≥1 valid sample that has canon_smiles
        # Matches benchmark_analysis_delinker/linkllama (exclude instances with no canon)
        success_with_canon = [
            r for r in success_instances if r.get("uniqueness_pct", 0) > 0
        ] if success_instances else []
        valid_pcts = [r["mol_valid_pct"] for r in success_with_canon]
        # Uniqueness: same filter
        # Matches benchmark_analysis_delinker/linkllama: n_valid = len(valid_canon_mol_smis)
        uniq_pcts = [r["uniqueness_pct"] for r in success_with_canon]

        mol_valid_mean = float(sum(valid_pcts) / len(valid_pcts)) if valid_pcts else 0.0
        mol_valid_std = float(statistics.stdev(valid_pcts)) if len(valid_pcts) >= 2 else 0.0
        uniq_mean = float(sum(uniq_pcts) / len(uniq_pcts)) if uniq_pcts else 0.0
        uniq_std = float(statistics.stdev(uniq_pcts)) if len(uniq_pcts) >= 2 else 0.0

        qed_means = [r["mean_mol_qed"] for r in success_with_canon if r.get("mean_mol_qed") is not None]
        sa_means = [r["mean_mol_sa"] for r in success_with_canon if r.get("mean_mol_sa") is not None]
        mean_qed = float(sum(qed_means) / len(qed_means)) if qed_means else None
        std_qed = float(statistics.stdev(qed_means)) if len(qed_means) >= 2 else (0.0 if qed_means else None)
        mean_sa = float(sum(sa_means) / len(sa_means)) if sa_means else None
        std_sa = float(statistics.stdev(sa_means)) if len(sa_means) >= 2 else (0.0 if sa_means else None)

        reason_means = [
            r["mean_pass_all_reasonability"]
            for r in success_with_canon
            if r.get("mean_pass_all_reasonability") is not None
        ]
        mean_pass_all_agg = float(sum(reason_means) / len(reason_means)) if reason_means else 0.0
        std_pass_all_agg = (
            float(statistics.stdev(reason_means))
            if len(reason_means) >= 2
            else (0.0 if reason_means else None)
        )

        # Novelty: only on folders with ≥1 valid sample (same as validity/uniqueness)
        novelty_pct_mean = novelty_pct_std = None
        success_names = {r["name"] for r in success_with_canon}
        if self._training_linkers_pkl:
            training_linkers = _load_training_linkers_pkl(self._training_linkers_pkl)
            if training_linkers is not None:
                novelty_per_instance: List[float] = []
                for inst in instances:
                    if inst.name not in success_names:
                        continue
                    results_inst = by_name.get(inst.name, [])
                    inst_linkers = [
                        r["canon_linker"]
                        for r in results_inst
                        if r.get("valid") and r.get("canon_linker")
                    ]
                    if inst_linkers:
                        unique_linkers = set(inst_linkers)
                        n_train = sum(1 for c in unique_linkers if c in training_linkers)
                        novelty_per_instance.append(
                            100.0 * (len(unique_linkers) - n_train) / len(unique_linkers)
                        )
                if novelty_per_instance:
                    novelty_pct_mean = float(sum(novelty_per_instance) / len(novelty_per_instance))
                    novelty_pct_std = (
                        float(statistics.stdev(novelty_per_instance))
                        if len(novelty_per_instance) >= 2
                        else 0.0
                    )

        # Geometry: per-instance RMSD list + energy delta (DeLinker-style fragment RMSD).
        # Each worker returns (inst_name, rmsd_list, ed_list) or None.
        geo_results: List[Tuple[str, List[float], List[float]]] = []
        if compute_geometry and self._geometry is not None and self._reference_sdfs_dir is not None:
            geo_tasks: List[Tuple[str, List[Tuple[str, Any]], str, str]] = []
            for inst in instances:
                valid_samples = [
                    s
                    for s in inst.samples
                    if sample_key_to_result.get((inst.name, s.sample_id), {}).get("valid")
                ]
                samples_to_use = (
                    valid_samples
                    if geometry_max_candidates is None
                    else valid_samples[: geometry_max_candidates]
                )
                samples_data = [(s.mol_repr_type, str(s.mol_repr)) for s in samples_to_use]
                if not samples_data:
                    continue
                geo_tasks.append(
                    (
                        inst.name,
                        samples_data,
                        str(self._csv_path),
                        str(self._reference_sdfs_dir),
                    )
                )
            if geo_tasks:
                if n_workers > 1:
                    with ProcessPoolExecutor(max_workers=n_workers) as ex:
                        futures = [ex.submit(_geometry_instance_worker, t) for t in geo_tasks]
                        for fut in tqdm(as_completed(futures), total=len(futures), desc="Geometry", leave=False):
                            summary = fut.result()
                            if summary is not None:
                                geo_results.append(summary)
                else:
                    for t in tqdm(geo_tasks, desc="Geometry", leave=False):
                        summary = _geometry_instance_worker(t)
                        if summary is not None:
                            geo_results.append(summary)

        n_geo = len(geo_results)
        mean_rmsd = std_rmsd = mean_best = std_best = mean_ed = std_ed = None
        ed_p25 = ed_p50 = ed_p75 = None
        if geo_results:
            geo_by_name: Dict[str, Dict[str, Any]] = {}
            all_rmsds: List[float] = []
            best_per_inst: List[float] = []
            ed_inst_means: List[float] = []
            for inst_name_g, rl, edl in geo_results:
                all_rmsds.extend(rl)
                best_per_inst.append(min(rl))
                mean_rmsd_i = float(sum(rl) / len(rl))
                std_rmsd_i = float(statistics.stdev(rl)) if len(rl) >= 2 else None
                best_rmsd_i = float(min(rl))
                if edl:
                    mean_ed_i = float(sum(edl) / len(edl))
                    std_ed_i = float(statistics.stdev(edl)) if len(edl) >= 2 else None
                    if not (isinstance(mean_ed_i, float) and math.isnan(mean_ed_i)):
                        ed_inst_means.append(mean_ed_i)
                else:
                    mean_ed_i = float("nan")
                    std_ed_i = None
                geo_by_name[inst_name_g] = {
                    "mean_rmsd": mean_rmsd_i,
                    "std_rmsd": std_rmsd_i,
                    "best_rmsd": best_rmsd_i,
                    "mean_energy_delta": mean_ed_i if edl else None,
                    "std_energy_delta": std_ed_i,
                }
            for row in per_instance:
                gstats = geo_by_name.get(row["name"])
                if gstats:
                    row.update(gstats)
                else:
                    row.setdefault("mean_rmsd", None)
                    row.setdefault("std_rmsd", None)
                    row.setdefault("best_rmsd", None)
                    row.setdefault("mean_energy_delta", None)
                    row.setdefault("std_energy_delta", None)
            if all_rmsds:
                mean_rmsd = float(sum(all_rmsds) / len(all_rmsds))
                std_rmsd = float(statistics.stdev(all_rmsds)) if len(all_rmsds) >= 2 else 0.0
            if best_per_inst:
                mean_best = float(sum(best_per_inst) / len(best_per_inst))
                std_best = float(statistics.stdev(best_per_inst)) if len(best_per_inst) >= 2 else 0.0
            ed = [m for m in ed_inst_means if not (isinstance(m, float) and math.isnan(m))]
            if ed:
                mean_ed = float(sum(ed) / len(ed))
                std_ed = float(statistics.stdev(ed)) if len(ed) >= 2 else 0.0
                sorted_ed = sorted(ed)
                n_ed = len(sorted_ed)
                ed_p25 = sorted_ed[int(0.25 * (n_ed - 1))] if n_ed > 0 else None
                ed_p50 = sorted_ed[int(0.50 * (n_ed - 1))] if n_ed > 0 else None
                ed_p75 = sorted_ed[int(0.75 * (n_ed - 1))] if n_ed > 0 else None
        else:
            for row in per_instance:
                row.setdefault("mean_rmsd", None)
                row.setdefault("std_rmsd", None)
                row.setdefault("best_rmsd", None)
                row.setdefault("mean_energy_delta", None)
                row.setdefault("std_energy_delta", None)

        return UnifiedBenchmarkResult(
            method=method,
            n_instances=n_instances,
            total_samples=total_samples,
            subfolders_failed=subfolders_failed,
            n_expected_instances=n_expected_instances,
            n_missing_instances=n_missing_instances,
            n_instances_zero_validity=n_instances_zero_validity,
            n_samples_invalid=n_samples_invalid,
            mol_valid_pct_mean=mol_valid_mean,
            mol_valid_pct_std=mol_valid_std,
            uniqueness_pct_mean=uniq_mean,
            uniqueness_pct_std=uniq_std,
            recovery_total_pct=recovery_total_pct,
            mean_mol_qed=mean_qed,
            std_mol_qed=std_qed,
            mean_mol_sa=mean_sa,
            std_mol_sa=std_sa,
            novelty_pct_mean=novelty_pct_mean,
            novelty_pct_std=novelty_pct_std,
            overall_success_rate_pct=overall_success_rate_pct,
            folder_success_rate_pct=folder_success_rate_pct,
            n_reasonability=n_reason,
            pct_pass_undesirable=100.0 * pass_undes / n_reason if n_reason > 0 else 0.0,
            pct_pass_pains=100.0 * pass_pains / n_reason if n_reason > 0 else 0.0,
            pct_pass_reos=100.0 * pass_reos / n_reason if n_reason > 0 else 0.0,
            pct_pass_bad_ring=100.0 * pass_bad / n_reason if n_reason > 0 else 0.0,
            pct_pass_problematic_ring=100.0 * pass_prob / n_reason if n_reason > 0 else 0.0,
            pct_pass_all_reasonability=mean_pass_all_agg,
            std_pct_pass_all_reasonability=std_pass_all_agg,
            n_geometry=n_geo,
            mean_rmsd=mean_rmsd,
            std_rmsd=std_rmsd,
            mean_best_rmsd=mean_best,
            std_best_rmsd=std_best,
            mean_energy_delta=mean_ed,
            std_energy_delta=std_ed,
            energy_delta_p25=ed_p25,
            energy_delta_p50=ed_p50,
            energy_delta_p75=ed_p75,
            per_instance=per_instance,
        )


def run_benchmark(
    method: str,
    csv_path: Path,
    input_path: Path,
    reference_sdfs_dir: Optional[Path] = None,
    training_linkers_pkl: Optional[Path] = None,
    n_workers: int = 1,
    n_per_instance: Optional[int] = 100,
    run_index: Optional[int] = None,
    expected_samples_per_instance: Optional[int] = 100,
    compute_geometry: bool = True,
    geometry_max_candidates: Optional[int] = None,
    max_instances: Optional[int] = None,
    linkllama_failed_cases_path: Optional[Path] = None,
) -> UnifiedBenchmarkResult:
    """
    Run the unified benchmark for one method.

    - method: "difflinker" | "delinker" | "linkllama"
    - csv_path: ground truth CSV (name, SMILES, fragments, total_energy)
    - input_path: DiffLinker output dir, DeLinker .smi or subdir, or LinkLlama .pkl
    - reference_sdfs_dir: dir with conformer_<name>.sdf (optional; geometry skipped if None)
    - training_linkers_pkl: optional; novelty (% of distinct canonical linkers not in training inventory)
    """
    csv_path = Path(csv_path)
    input_path = Path(input_path)
    bench = UnifiedBenchmark(
        csv_path,
        reference_sdfs_dir=reference_sdfs_dir,
        training_linkers_pkl=training_linkers_pkl,
    )

    if method == "difflinker":
        from . import source_difflinker
        instances = source_difflinker.load_instances(
            input_dir=input_path,
            name_to_fragments=bench.name_to_fragments,
            name_to_gt_smiles=bench.name_to_gt_smiles,
            max_samples_per_instance=n_per_instance,
        )
    elif method == "delinker":
        from . import source_delinker
        instances = source_delinker.load_instances(
            input_path=input_path,
            name_to_fragments=bench.name_to_fragments,
            name_to_gt_smiles=bench.name_to_gt_smiles,
            expected_samples_per_instance=expected_samples_per_instance,
            max_samples_per_instance=n_per_instance,
        )
    elif method == "linkllama":
        from . import source_linkllama
        instances = source_linkllama.load_instances(
            pkl_path=input_path,
            name_to_fragments=bench.name_to_fragments,
            name_to_gt_smiles=bench.name_to_gt_smiles,
            run_index=run_index,
            max_samples_per_instance=n_per_instance,
            failed_cases_path=linkllama_failed_cases_path,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if max_instances is not None:
        instances = instances[:max_instances]

    return bench.run(
        method=method,
        instances=instances,
        n_workers=n_workers,
        geometry_max_candidates=geometry_max_candidates,
        compute_geometry=compute_geometry,
        expected_samples_per_instance=expected_samples_per_instance,
    )


def run_all_benchmarks(
    csv_path: Path,
    reference_sdfs_dir: Optional[Path],
    difflinker_dir: Optional[Path] = None,
    delinker_path: Optional[Path] = None,
    linkllama_pkl: Optional[Path] = None,
    training_linkers_pkl: Optional[Path] = None,
    n_workers: int = 1,
    n_per_instance: Optional[int] = 1,
    run_index: Optional[int] = None,
    expected_samples_per_instance: Optional[int] = 100,
    compute_geometry: bool = True,
    geometry_max_candidates: Optional[int] = None,
    max_instances: Optional[int] = None,
    linkllama_failed_cases_path: Optional[Path] = None,
) -> Dict[str, UnifiedBenchmarkResult]:
    """
    Run the unified benchmark for each provided inference result.
    Returns dict method -> UnifiedBenchmarkResult (only for methods whose path was given).
    """
    results: Dict[str, UnifiedBenchmarkResult] = {}
    if difflinker_dir is not None:
        results["difflinker"] = run_benchmark(
            "difflinker",
            csv_path=csv_path,
            input_path=difflinker_dir,
            reference_sdfs_dir=reference_sdfs_dir,
            training_linkers_pkl=training_linkers_pkl,
            n_workers=n_workers,
            n_per_instance=n_per_instance,
            compute_geometry=compute_geometry,
            geometry_max_candidates=geometry_max_candidates,
            max_instances=max_instances,
        )
    if delinker_path is not None:
        results["delinker"] = run_benchmark(
            "delinker",
            csv_path=csv_path,
            input_path=delinker_path,
            reference_sdfs_dir=reference_sdfs_dir,
            training_linkers_pkl=training_linkers_pkl,
            n_workers=n_workers,
            n_per_instance=n_per_instance,
            expected_samples_per_instance=expected_samples_per_instance,
            compute_geometry=compute_geometry,
            geometry_max_candidates=geometry_max_candidates,
            max_instances=max_instances,
        )
    if linkllama_pkl is not None:
        results["linkllama"] = run_benchmark(
            "linkllama",
            csv_path=csv_path,
            input_path=linkllama_pkl,
            reference_sdfs_dir=reference_sdfs_dir,
            training_linkers_pkl=training_linkers_pkl,
            n_workers=n_workers,
            n_per_instance=n_per_instance,
            run_index=run_index,
            expected_samples_per_instance=expected_samples_per_instance,
            compute_geometry=compute_geometry,
            geometry_max_candidates=geometry_max_candidates,
            max_instances=max_instances,
            linkllama_failed_cases_path=linkllama_failed_cases_path,
        )
    return results


def result_to_csv_row(r: UnifiedBenchmarkResult, dataset: str) -> Dict[str, Any]:
    """Flatten UnifiedBenchmarkResult to a dict for CSV export."""
    return {
        "dataset": dataset,
        "method": r.method,
        "n_instances": r.n_instances,
        "n_expected_instances": r.n_expected_instances,
        "n_missing_instances": r.n_missing_instances,
        "n_instances_zero_validity": r.n_instances_zero_validity,
        "n_samples_invalid": r.n_samples_invalid,
        "total_samples": r.total_samples,
        "mol_valid_pct_mean": r.mol_valid_pct_mean,
        "mol_valid_pct_std": r.mol_valid_pct_std,
        "uniqueness_pct_mean": r.uniqueness_pct_mean,
        "uniqueness_pct_std": r.uniqueness_pct_std,
        "recovery_total_pct": r.recovery_total_pct,
        "mean_mol_qed": r.mean_mol_qed,
        "std_mol_qed": r.std_mol_qed,
        "mean_mol_sa": r.mean_mol_sa,
        "std_mol_sa": r.std_mol_sa,
        "novelty_pct_mean": r.novelty_pct_mean,
        "novelty_pct_std": r.novelty_pct_std,
        "overall_success_rate_pct": r.overall_success_rate_pct,
        "folder_success_rate_pct": r.folder_success_rate_pct,
        "pct_pass_undesirable": r.pct_pass_undesirable,
        "pct_pass_pains": r.pct_pass_pains,
        "pct_pass_reos": r.pct_pass_reos,
        "pct_pass_bad_ring": r.pct_pass_bad_ring,
        "pct_pass_problematic_ring": r.pct_pass_problematic_ring,
        "pct_pass_all_reasonability": r.pct_pass_all_reasonability,
        "std_pct_pass_all_reasonability": r.std_pct_pass_all_reasonability,
        "n_geometry": r.n_geometry,
        "mean_rmsd": r.mean_rmsd,
        "std_rmsd": r.std_rmsd,
        "mean_best_rmsd": r.mean_best_rmsd,
        "std_best_rmsd": r.std_best_rmsd,
        "mean_energy_delta": r.mean_energy_delta,
        "std_energy_delta": r.std_energy_delta,
        "energy_delta_p25": r.energy_delta_p25,
        "energy_delta_p50": r.energy_delta_p50,
        "energy_delta_p75": r.energy_delta_p75,
    }


def per_instance_rows_to_dataframe(r: UnifiedBenchmarkResult, dataset: str) -> pd.DataFrame:
    """One row per instance (folder) with dataset and method columns for CSV export."""
    rows = [{**row, "dataset": dataset, "method": r.method} for row in r.per_instance]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reduced benchmark (merged from unified_benchmark_reduced.py)
# Validity, uniqueness, recovery, QED, SA, novelty only.
# No geometry (RMSD/energy) or reasonability metrics.
# ---------------------------------------------------------------------------


def _sample_worker_reduced(args: Tuple[GeneratedSample, int]) -> Tuple[int, Dict[str, Any]]:
    """Lightweight sample worker without reasonability metrics."""
    sample, pos = args
    out: Dict[str, Any] = {
        "name": sample.name,
        "sample_id": sample.sample_id,
        "valid": False,
        "canon_smiles": None,
        "canon_linker": None,
        "qed": None,
        "sa": None,
        "recovered": False,
    }
    try:
        if sample.mol_repr_type == "smiles":
            smi = str(sample.mol_repr).strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                return pos, out
        else:
            sdf_path = Path(sample.mol_repr)
            supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            mol0 = next(iter(supp), None)
            if mol0 is None:
                return pos, out
            mol = Chem.RemoveHs(mol0)
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
            except Exception:
                return pos, out

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return pos, out

        clean_frag = build_clean_frag_charged(sample.fragments_smi)
        if clean_frag is None or len(mol.GetSubstructMatches(clean_frag)) == 0:
            return pos, out

        out["valid"] = True
        canon = mol_to_compact_canonical_smiles(mol)
        out["canon_smiles"] = canon
        if sample.linker_smiles:
            out["canon_linker"] = canonicalize_linker(sample.linker_smiles)
        elif canon:
            try:
                mol_l = Chem.MolFromSmiles(canon)
                if mol_l is not None:
                    Chem.SanitizeMol(mol_l)
                    raw_ls = get_linker(mol_l, clean_frag, sample.fragments_smi)
                    if raw_ls:
                        out["canon_linker"] = canonicalize_linker(raw_ls)
            except Exception:
                pass
        out["qed"] = calculate_qed(mol)
        out["sa"] = calculate_sa(mol)

        gt_canon = canonicalize_molecule(sample.gt_smiles) if sample.gt_smiles else None
        out["recovered"] = bool(gt_canon and canon and gt_canon == canon)
        return pos, out
    except Exception:
        return pos, out


@dataclass
class UnifiedBenchmarkResultReduced:
    method: str
    n_instances: int
    n_expected_instances: int
    n_missing_instances: int
    n_instances_zero_validity: int
    n_samples_invalid: int
    total_samples: int
    mol_valid_pct_mean: float
    mol_valid_pct_std: float
    uniqueness_pct_mean: float
    uniqueness_pct_std: float
    recovery_total_pct: float
    mean_mol_qed: Optional[float]
    std_mol_qed: Optional[float]
    mean_mol_sa: Optional[float]
    std_mol_sa: Optional[float]
    novelty_pct_mean: Optional[float]
    novelty_pct_std: Optional[float]
    overall_success_rate_pct: Optional[float]
    folder_success_rate_pct: Optional[float]


class UnifiedBenchmarkReduced:
    """Benchmark: validity, uniqueness, recovery, QED, SA, novelty. No geometry/reasonability."""

    def __init__(
        self,
        csv_path: Path,
        training_linkers_pkl: Optional[Path] = None,
    ) -> None:
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        self.name_to_fragments: Dict[str, str] = {}
        self.name_to_gt_smiles: Dict[str, str] = {}
        for _, row in df.iterrows():
            name = row.get("name")
            if pd.isna(name):
                continue
            name = str(name).strip()
            frags = row.get("fragments", "")
            if pd.notna(frags):
                self.name_to_fragments[name] = str(frags).strip()
            if "SMILES" in row and pd.notna(row.get("SMILES")):
                self.name_to_gt_smiles[name] = str(row["SMILES"]).strip()
        self._csv_path = csv_path
        self._training_linkers_pkl = Path(training_linkers_pkl) if training_linkers_pkl else None

    def run(
        self,
        method: str,
        instances: List[InstanceSamples],
        n_workers: int = 1,
        expected_samples_per_instance: Optional[int] = None,
    ) -> UnifiedBenchmarkResultReduced:
        flat: List[GeneratedSample] = []
        for inst in instances:
            flat.extend(inst.samples)
        total_samples = (
            len(instances) * expected_samples_per_instance
            if expected_samples_per_instance is not None
            else len(flat)
        )

        per_sample_by_pos: List[Optional[Dict[str, Any]]] = [None] * len(flat)
        tasks = [(s, i) for i, s in enumerate(flat)]
        if n_workers > 1 and tasks:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_sample_worker_reduced, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Props"):
                    pos, res = fut.result()
                    per_sample_by_pos[pos] = res
        else:
            for t in tqdm(tasks, desc="Props"):
                pos, res = _sample_worker_reduced(t)
                per_sample_by_pos[pos] = res

        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for i in range(len(flat)):
            res = per_sample_by_pos[i]
            if res is not None:
                by_name.setdefault(flat[i].name, []).append(res)

        per_instance: List[Dict[str, Any]] = []
        success_instances: List[Dict[str, Any]] = []
        recovered_success = 0

        for inst in instances:
            results = by_name.get(inst.name, [])
            denom = inst.expected_total if inst.expected_total is not None else len(results)
            valid = [r for r in results if r.get("valid")]
            n_valid = len(valid)
            mol_valid_pct = 100.0 * n_valid / denom if denom and denom > 0 else 0.0
            canon_list = [r["canon_smiles"] for r in valid if r.get("canon_smiles")]
            uniqueness_pct = 100.0 * len(set(canon_list)) / len(canon_list) if canon_list else 0.0
            recovered = any(r.get("recovered") for r in valid)
            qeds = [r["qed"] for r in valid if r.get("qed") is not None]
            sas = [r["sa"] for r in valid if r.get("sa") is not None]
            mean_qed = float(sum(qeds) / len(qeds)) if qeds else None
            mean_sa = float(sum(sas) / len(sas)) if sas else None

            row = {
                "name": inst.name,
                "total_samples": denom if denom is not None else len(results),
                "n_valid": n_valid,
                "mol_valid_pct": mol_valid_pct,
                "uniqueness_pct": uniqueness_pct,
                "recovered": recovered,
                "mean_mol_qed": mean_qed,
                "mean_mol_sa": mean_sa,
            }
            per_instance.append(row)
            if mol_valid_pct > 0:
                success_instances.append(row)
                if recovered:
                    recovered_success += 1

        n_instances = len(per_instance)
        n_success = len(success_instances)
        subfolders_failed = n_instances - n_success
        n_expected_instances = len(self.name_to_fragments)
        n_missing_instances = max(0, n_expected_instances - n_instances)
        n_instances_zero_validity = subfolders_failed
        total_valid_samples = sum(r["n_valid"] for r in per_instance)
        n_samples_invalid = total_samples - total_valid_samples
        recovery_total_pct = (
            100.0 * recovered_success / n_expected_instances
            if n_expected_instances and n_expected_instances > 0
            else 0.0
        )
        overall_success_rate_pct = (
            100.0 * total_valid_samples / total_samples if total_samples > 0 else None
        )
        folder_success_rate_pct = (
            100.0 * n_success / n_expected_instances
            if n_expected_instances and n_expected_instances > 0
            else None
        )

        success_with_canon = [
            r for r in success_instances if r.get("uniqueness_pct", 0) > 0
        ] if success_instances else []
        valid_pcts = [r["mol_valid_pct"] for r in success_with_canon]
        uniq_pcts = [r["uniqueness_pct"] for r in success_with_canon]
        mol_valid_mean = float(sum(valid_pcts) / len(valid_pcts)) if valid_pcts else 0.0
        mol_valid_std = float(statistics.stdev(valid_pcts)) if len(valid_pcts) >= 2 else 0.0
        uniq_mean = float(sum(uniq_pcts) / len(uniq_pcts)) if uniq_pcts else 0.0
        uniq_std = float(statistics.stdev(uniq_pcts)) if len(uniq_pcts) >= 2 else 0.0

        qed_means = [r["mean_mol_qed"] for r in success_with_canon if r.get("mean_mol_qed") is not None]
        sa_means = [r["mean_mol_sa"] for r in success_with_canon if r.get("mean_mol_sa") is not None]
        mean_qed = float(sum(qed_means) / len(qed_means)) if qed_means else None
        std_qed = float(statistics.stdev(qed_means)) if len(qed_means) >= 2 else (0.0 if qed_means else None)
        mean_sa = float(sum(sa_means) / len(sa_means)) if sa_means else None
        std_sa = float(statistics.stdev(sa_means)) if len(sa_means) >= 2 else (0.0 if sa_means else None)

        novelty_pct_mean = novelty_pct_std = None
        success_names = {r["name"] for r in success_with_canon}
        if self._training_linkers_pkl:
            training_linkers = _load_training_linkers_pkl(self._training_linkers_pkl)
            if training_linkers is not None:
                novelty_per_instance: List[float] = []
                for inst in instances:
                    if inst.name not in success_names:
                        continue
                    results_inst = by_name.get(inst.name, [])
                    inst_linkers = [
                        r["canon_linker"]
                        for r in results_inst
                        if r.get("valid") and r.get("canon_linker")
                    ]
                    if inst_linkers:
                        unique_linkers = set(inst_linkers)
                        n_train = sum(1 for c in unique_linkers if c in training_linkers)
                        novelty_per_instance.append(
                            100.0 * (len(unique_linkers) - n_train) / len(unique_linkers)
                        )
                if novelty_per_instance:
                    novelty_pct_mean = float(sum(novelty_per_instance) / len(novelty_per_instance))
                    novelty_pct_std = (
                        float(statistics.stdev(novelty_per_instance))
                        if len(novelty_per_instance) >= 2
                        else 0.0
                    )

        return UnifiedBenchmarkResultReduced(
            method=method,
            n_instances=n_instances,
            n_expected_instances=n_expected_instances,
            n_missing_instances=n_missing_instances,
            n_instances_zero_validity=n_instances_zero_validity,
            n_samples_invalid=n_samples_invalid,
            total_samples=total_samples,
            mol_valid_pct_mean=mol_valid_mean,
            mol_valid_pct_std=mol_valid_std,
            uniqueness_pct_mean=uniq_mean,
            uniqueness_pct_std=uniq_std,
            recovery_total_pct=recovery_total_pct,
            mean_mol_qed=mean_qed,
            std_mol_qed=std_qed,
            mean_mol_sa=mean_sa,
            std_mol_sa=std_sa,
            novelty_pct_mean=novelty_pct_mean,
            novelty_pct_std=novelty_pct_std,
            overall_success_rate_pct=overall_success_rate_pct,
            folder_success_rate_pct=folder_success_rate_pct,
        )


def run_benchmark_reduced(
    method: str,
    csv_path: Path,
    input_path: Path,
    training_linkers_pkl: Optional[Path] = None,
    n_workers: int = 1,
    n_per_instance: Optional[int] = 100,
    run_index: Optional[int] = None,
    expected_samples_per_instance: Optional[int] = 100,
    max_instances: Optional[int] = None,
    linkllama_failed_cases_path: Optional[Path] = None,
) -> UnifiedBenchmarkResultReduced:
    csv_path = Path(csv_path)
    input_path = Path(input_path)
    bench = UnifiedBenchmarkReduced(csv_path, training_linkers_pkl=training_linkers_pkl)

    if method == "difflinker":
        from . import source_difflinker
        instances = source_difflinker.load_instances(
            input_dir=input_path,
            name_to_fragments=bench.name_to_fragments,
            name_to_gt_smiles=bench.name_to_gt_smiles,
            max_samples_per_instance=n_per_instance,
        )
    elif method == "delinker":
        from . import source_delinker
        instances = source_delinker.load_instances(
            input_path=input_path,
            name_to_fragments=bench.name_to_fragments,
            name_to_gt_smiles=bench.name_to_gt_smiles,
            expected_samples_per_instance=expected_samples_per_instance,
            max_samples_per_instance=n_per_instance,
        )
    elif method == "linkllama":
        from . import source_linkllama
        instances = source_linkllama.load_instances(
            pkl_path=input_path,
            name_to_fragments=bench.name_to_fragments,
            name_to_gt_smiles=bench.name_to_gt_smiles,
            run_index=run_index,
            max_samples_per_instance=n_per_instance,
            failed_cases_path=linkllama_failed_cases_path,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if max_instances is not None:
        instances = instances[:max_instances]

    return bench.run(
        method=method,
        instances=instances,
        n_workers=n_workers,
        expected_samples_per_instance=expected_samples_per_instance,
    )


def result_reduced_to_csv_row(r: UnifiedBenchmarkResultReduced, dataset: str) -> Dict[str, Any]:
    """Flatten UnifiedBenchmarkResultReduced to a dict for CSV export."""
    return {
        "dataset": dataset,
        "method": r.method,
        "n_instances": r.n_instances,
        "n_expected_instances": r.n_expected_instances,
        "n_missing_instances": r.n_missing_instances,
        "n_instances_zero_validity": r.n_instances_zero_validity,
        "n_samples_invalid": r.n_samples_invalid,
        "total_samples": r.total_samples,
        "mol_valid_pct_mean": r.mol_valid_pct_mean,
        "mol_valid_pct_std": r.mol_valid_pct_std,
        "uniqueness_pct_mean": r.uniqueness_pct_mean,
        "uniqueness_pct_std": r.uniqueness_pct_std,
        "recovery_total_pct": r.recovery_total_pct,
        "mean_mol_qed": r.mean_mol_qed,
        "std_mol_qed": r.std_mol_qed,
        "mean_mol_sa": r.mean_mol_sa,
        "std_mol_sa": r.std_mol_sa,
        "novelty_pct_mean": r.novelty_pct_mean,
        "novelty_pct_std": r.novelty_pct_std,
        "overall_success_rate_pct": r.overall_success_rate_pct,
        "folder_success_rate_pct": r.folder_success_rate_pct,
    }


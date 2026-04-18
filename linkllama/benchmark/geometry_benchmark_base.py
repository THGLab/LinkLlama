#!/usr/bin/env python3
"""
Shared base for geometry-only benchmark (energy + RMSD).

GeometryBenchmark: validity (fragments + protonation) first, then RMSD/energy
only for valid samples. Fragment RMSD: align on both fragments, single spyrmsd over combined atoms
(DeLinker GetBestRMS semantics). Uses atom indices + AlignMol(atomMap) so it works
for DiffLinker where get_frags fails. Requires spyrmsd.

Reports: mean_rmsd (avg of all sample RMSDs), mean_best_rmsd (avg of min per instance).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign

RDLogger.DisableLog("rdApp.*")

try:
    from spyrmsd.rmsd import rmsd as _spyrmsd_rmsd_fn
except ImportError:
    _spyrmsd_rmsd_fn = None


@dataclass
class EnergyRMSDResult:
    """Per-sample geometry result (DeLinker-style fragment RMSD)."""

    name: str
    rmsd: float
    mmff_energy_after: float
    ref_energy: float
    energy_delta: float


def normalize_fragment_smiles(smi: str) -> str:
    """Convert [*:1]/[*:2] to plain *."""
    return re.sub(r"\[\*:[12]\]", "*", smi) if smi else smi


def build_clean_frag_charged(fragments_smi: str) -> Optional[Chem.Mol]:
    """Build clean frag (H at dummies) with charges preserved."""
    try:
        fragments_norm = normalize_fragment_smiles(fragments_smi)
        du = Chem.MolFromSmiles("*")
        clean = Chem.RemoveHs(
            AllChem.ReplaceSubstructs(
                Chem.MolFromSmiles(fragments_norm),
                du,
                Chem.MolFromSmiles("[H]"),
                True,
            )[0]
        )
        return clean
    except Exception:
        return None


def get_frag_atom_indices(
    full_mol: Chem.Mol, clean_frag: Chem.Mol
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Return (frag1_indices, frag2_indices) via GetSubstructMatches(clean_frag) only.
    No fallback; used for fragment RMSD so gen and ref use the same matching.
    """
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    if not matches:
        return None
    match = matches[0]
    frags = Chem.rdmolops.GetMolFrags(clean_frag, asMols=False)
    if len(frags) != 2:
        return None
    n1, n2 = len(frags[0]), len(frags[1])
    frag1_idx = list(match[:n1])
    frag2_idx = list(match[n1 : n1 + n2])
    return (frag1_idx, frag2_idx)


def _get_coords_and_atomic_nums(
    mol: Chem.Mol, atom_indices: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords Nx3, atomic_numbers) for given atom indices."""
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in atom_indices], dtype=float)
    an = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in atom_indices], dtype=np.int32)
    return coords, an


def _align_on_both_fragments_and_compute_rmsd(
    gen_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    clean_frag: Chem.Mol,
) -> Optional[float]:
    """
    Align gen_mol to ref_mol on BOTH fragments, then compute single RMSD over
    all fragment atoms via spyrmsd. Matches DeLinker GetBestRMS semantics.
    """
    if _spyrmsd_rmsd_fn is None:
        return None
    ref_indices = get_frag_atom_indices(ref_mol, clean_frag)
    gen_indices = get_frag_atom_indices(gen_mol, clean_frag)
    if ref_indices is None or gen_indices is None:
        return None
    ref_f1_idx, ref_f2_idx = ref_indices
    gen_f1_idx, gen_f2_idx = gen_indices

    atom_map = (
        [(gen_f1_idx[i], ref_f1_idx[i]) for i in range(len(ref_f1_idx))]
        + [(gen_f2_idx[i], ref_f2_idx[i]) for i in range(len(ref_f2_idx))]
    )
    try:
        rdMolAlign.AlignMol(gen_mol, ref_mol, atomMap=atom_map)
    except Exception:
        return None

    gen_coords = np.vstack([
        _get_coords_and_atomic_nums(gen_mol, gen_f1_idx)[0],
        _get_coords_and_atomic_nums(gen_mol, gen_f2_idx)[0],
    ])
    ref_coords = np.vstack([
        _get_coords_and_atomic_nums(ref_mol, ref_f1_idx)[0],
        _get_coords_and_atomic_nums(ref_mol, ref_f2_idx)[0],
    ])
    gen_an = np.concatenate([
        _get_coords_and_atomic_nums(gen_mol, gen_f1_idx)[1],
        _get_coords_and_atomic_nums(gen_mol, gen_f2_idx)[1],
    ])
    ref_an = np.concatenate([
        _get_coords_and_atomic_nums(ref_mol, ref_f1_idx)[1],
        _get_coords_and_atomic_nums(ref_mol, ref_f2_idx)[1],
    ])

    if gen_an.shape != ref_an.shape or not np.all(gen_an == ref_an):
        return None

    return float(
        _spyrmsd_rmsd_fn(
            gen_coords, ref_coords, gen_an, ref_an, center=True, minimize=True
        )
    )


def get_frags(full_mol: Chem.Mol, clean_frag: Chem.Mol, fragments_smi: Optional[str] = None) -> Optional[Chem.Mol]:
    """
    DeLinker-style: remove linker from full_mol, return fragments-only mol.
    full_mol must contain clean_frag; linker = full - frag atoms.
    Returns mol with only the two fragments (linker removed), or None.

    If GetSubstructMatches(clean_frag) fails (e.g. SDF-derived bracket SMILES),
    falls back to matching each fragment part separately via fragments_smi.
    """
    try:
        matches = list(full_mol.GetSubstructMatches(clean_frag))
        linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
        if linker_len == 0:
            return Chem.RWMol(full_mol).GetMol()
        if not matches and fragments_smi:
            frag_match = _get_frag_indices_by_parts(full_mol, fragments_smi)
            if frag_match is not None:
                linker_atoms = set(range(full_mol.GetNumHeavyAtoms())) - set(frag_match)
                mol_rw = Chem.RWMol(full_mol)
                for idx in sorted(linker_atoms, reverse=True):
                    mol_rw.RemoveAtom(idx)
                return Chem.Mol(mol_rw)
            return None
        work = Chem.RWMol(full_mol)
        Chem.Kekulize(work, clearAromaticFlags=True)
        work_mol = work.GetMol()
        if not matches:
            return None
        all_frags: List[Chem.Mol] = []
        all_frags_lengths: List[int] = []
        for match in matches:
            mol_rw = Chem.RWMol(work_mol)
            linker_atoms = set(range(work_mol.GetNumHeavyAtoms())) - set(match)
            for idx in sorted(match, reverse=True):
                mol_rw.RemoveAtom(idx)
            linker = Chem.Mol(mol_rw)
            if linker.GetNumHeavyAtoms() == linker_len:
                mol_rw2 = Chem.RWMol(work_mol)
                for idx in sorted(linker_atoms, reverse=True):
                    mol_rw2.RemoveAtom(idx)
                frags = Chem.Mol(mol_rw2)
                all_frags.append(frags)
                frag_frags = Chem.rdmolops.GetMolFrags(frags, asMols=False)
                all_frags_lengths.append(len(frag_frags))
                if len(frag_frags) == 2:
                    return frags
        if not all_frags:
            return None
        return all_frags[int(np.argmax(all_frags_lengths))]
    except Exception:
        return None


def _get_frag_indices_by_parts(full_mol: Chem.Mol, fragments_smi: str) -> Optional[List[int]]:
    """
    Match each fragment part separately (neutral query for SDF compatibility);
    return combined match indices, or None.
    """
    try:
        parts = fragments_smi.strip().split(".")
        if len(parts) != 2:
            return None
        idx1: Optional[Tuple[int, ...]] = None
        idx2: Optional[Tuple[int, ...]] = None
        for part in parts:
            part = part.strip()
            q_smi = part.replace("[*:1]", "[H]").replace("[*:2]", "[H]")
            q_mol = Chem.MolFromSmiles(q_smi)
            if q_mol is None:
                return None
            for atom in q_mol.GetAtoms():
                atom.SetFormalCharge(0)
            matches = list(full_mol.GetSubstructMatches(q_mol))
            if not matches:
                return None
            if idx1 is None:
                idx1 = matches[0]
            else:
                idx2 = matches[0]
        if idx1 is None or idx2 is None:
            return None
        combined = set(idx1) | set(idx2)
        if len(combined) != len(idx1) + len(idx2):
            return None
        return list(combined)
    except Exception:
        return None


class GeometryBenchmark:
    """
    Geometry benchmark: validity (fragments + protonation) first, then RMSD/energy
    only for valid samples. DeLinker-style fragment RMSD.
    """

    def __init__(self, csv_path: Path, reference_sdfs_dir: Path) -> None:
        csv_path = Path(csv_path)
        reference_sdfs_dir = Path(reference_sdfs_dir)
        self._name_to_fragments: Dict[str, str] = {}
        self._name_to_energy: Dict[str, float] = {}
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            name = row.get("name")
            if pd.isna(name):
                continue
            name = str(name).strip()
            frags = row.get("fragments", "")
            if pd.notna(frags):
                self._name_to_fragments[name] = str(frags).strip()
            en = row.get("total_energy", float("nan"))
            self._name_to_energy[name] = float(pd.to_numeric(en, errors="coerce"))
        self._reference_sdfs_dir = reference_sdfs_dir

    def is_valid(self, gen_mol: Chem.Mol, name: str) -> bool:
        """Validity: fragments exist (neutral query for SDF compatibility)."""
        fragments_smi = self._name_to_fragments.get(name)
        if not fragments_smi:
            return False
        parts = fragments_smi.strip().split(".")
        if len(parts) != 2:
            return False
        for part in parts:
            part = part.strip()
            q_smi = part.replace("[*:1]", "[H]").replace("[*:2]", "[H]")
            q_mol = Chem.MolFromSmiles(q_smi)
            if q_mol is None:
                return False
            for atom in q_mol.GetAtoms():
                atom.SetFormalCharge(0)
            if not gen_mol.GetSubstructMatches(q_mol):
                return False
        return True

    def _load_reference(self, name: str) -> Optional[Tuple[Chem.Mol, float]]:
        fragments_smi = self._name_to_fragments.get(name)
        ref_energy = self._name_to_energy.get(name, float("nan"))
        if not fragments_smi:
            return None
        ref_sdf = self._reference_sdfs_dir / f"conformer_{name}.sdf"
        if not ref_sdf.exists():
            return None
        supp = Chem.SDMolSupplier(str(ref_sdf), removeHs=False)
        ref_mol = Chem.RemoveHs(Chem.Mol(next((m for m in supp if m), None)))
        if ref_mol is None:
            return None
        clean_frag = build_clean_frag_charged(fragments_smi)
        if clean_frag is None or get_frag_atom_indices(ref_mol, clean_frag) is None:
            return None
        return (ref_mol, ref_energy)

    def _relax_mol(self, gen_mol: Chem.Mol) -> Tuple[Chem.Mol, float]:
        mol = Chem.RemoveHs(gen_mol)
        mol = Chem.AddHs(mol, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props else None
        AllChem.MMFFOptimizeMolecule(mol)
        e = float(ff.CalcEnergy()) if ff else float("nan")
        return Chem.RemoveHs(Chem.Mol(mol)), e

    def _relax_smiles(self, smiles: str) -> Tuple[Optional[Chem.Mol], float]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, float("nan")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            props = AllChem.MMFFGetMoleculeProperties(mol)
            ff = AllChem.MMFFGetMoleculeForceField(mol, props) if props else None
            AllChem.MMFFOptimizeMolecule(mol)
            e = float(ff.CalcEnergy()) if ff else float("nan")
            return Chem.RemoveHs(Chem.Mol(mol)), e
        except Exception:
            return None, float("nan")

    def _fragment_rmsd(
        self, gen_mol: Chem.Mol, ref_mol: Chem.Mol, clean_frag: Chem.Mol, fragments_smi: str
    ) -> Optional[float]:
        """
        Fragment RMSD: align on BOTH fragments, single spyrmsd over combined atoms.
        Matches DeLinker GetBestRMS semantics. Requires spyrmsd.
        Works for DiffLinker where get_frags/GetBestRMS fails.
        """
        return _align_on_both_fragments_and_compute_rmsd(gen_mol, ref_mol, clean_frag)

    def process_mol(self, gen_mol: Chem.Mol, name: str) -> Optional[EnergyRMSDResult]:
        if not self.is_valid(gen_mol, name):
            return None
        ref_data = self._load_reference(name)
        if ref_data is None:
            return None
        ref_mol, ref_energy = ref_data
        fragments_smi = self._name_to_fragments[name]
        clean_frag = build_clean_frag_charged(fragments_smi)
        if clean_frag is None:
            return None
        gen_relaxed, e_after = self._relax_mol(gen_mol)
        rmsd = self._fragment_rmsd(gen_relaxed, ref_mol, clean_frag, fragments_smi)
        if rmsd is None:
            return None
        energy_delta = (
            e_after - ref_energy
            if not np.isnan(ref_energy) and not np.isnan(e_after)
            else float("nan")
        )
        return EnergyRMSDResult(
            name=name,
            rmsd=rmsd,
            mmff_energy_after=e_after,
            ref_energy=ref_energy,
            energy_delta=energy_delta,
        )

    def process_smiles(self, smiles: str, name: str) -> Optional[EnergyRMSDResult]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if not self.is_valid(mol, name):
            return None
        gen_relaxed, e_after = self._relax_smiles(smiles)
        if gen_relaxed is None:
            return None
        ref_data = self._load_reference(name)
        if ref_data is None:
            return None
        ref_mol, ref_energy = ref_data
        fragments_smi = self._name_to_fragments.get(name)
        if not fragments_smi:
            return None
        clean_frag = build_clean_frag_charged(fragments_smi)
        if clean_frag is None:
            return None
        rmsd = self._fragment_rmsd(gen_relaxed, ref_mol, clean_frag, fragments_smi)
        if rmsd is None:
            return None
        energy_delta = (
            e_after - ref_energy
            if not np.isnan(ref_energy) and not np.isnan(e_after)
            else float("nan")
        )
        return EnergyRMSDResult(
            name=name,
            rmsd=rmsd,
            mmff_energy_after=e_after,
            ref_energy=ref_energy,
            energy_delta=energy_delta,
        )


def save_energy_rmsd_results(
    results: List[EnergyRMSDResult],
    save_prefix: Path,
) -> None:
    save_prefix = Path(save_prefix)
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in results]
    rmsds = np.array([r.rmsd for r in results], dtype=float)
    e = np.array([r.mmff_energy_after for r in results], dtype=float)
    ref_e = np.array([r.ref_energy for r in results], dtype=float)
    delta = np.array([r.energy_delta for r in results], dtype=float)
    np.save(f"{save_prefix}_names", np.array(names, dtype=object))
    np.save(f"{save_prefix}_rmsd", rmsds)
    np.save(f"{save_prefix}_mmff_energy_after", e)
    np.save(f"{save_prefix}_ref_energy", ref_e)
    np.save(f"{save_prefix}_energy_delta", delta)
    pd.DataFrame(
        {
            "name": names,
            "rmsd": rmsds,
            "mmff_energy_after": e,
            "ref_energy": ref_e,
            "energy_delta": delta,
        }
    ).to_csv(f"{save_prefix}_energy_rmsd.csv", index=False)
    print(f"[geometry_benchmark] Saved to {save_prefix}_energy_rmsd.csv")


def compute_summary(results: List[EnergyRMSDResult]) -> dict:
    if not results:
        return {"mean_rmsd": None, "mean_energy_delta": None, "n": 0}
    rmsds = np.array([r.rmsd for r in results], dtype=float)
    delta = np.array([r.energy_delta for r in results], dtype=float)
    valid_delta = delta[~np.isnan(delta)]
    return {
        "mean_rmsd": float(np.mean(rmsds)),
        "std_rmsd": float(np.std(rmsds)),
        "mean_energy_delta": float(np.mean(valid_delta)) if len(valid_delta) > 0 else float("nan"),
        "std_energy_delta": float(np.std(valid_delta)) if len(valid_delta) > 0 else float("nan"),
        "n": len(results),
    }


def load_csv_ground_truth(
    csv_path: Path,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Load CSV; return name -> fragments_smi, name -> total_energy."""
    bench = GeometryBenchmark(csv_path, Path("."))
    return bench._name_to_fragments, bench._name_to_energy

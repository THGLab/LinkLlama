#!/usr/bin/env python3
"""
Utility functions for UniDock benchmarking.

Includes:
- Molecular preparation (SMILES -> 3D SDF/PDBQT, SDF -> PDBQT)
- Docking box calculation from reference ligand SDF
- Receptor preparation
- PDBQT affinity parsing and PDBQT -> SDF conversion
- Valid SMILES extraction from generated samples (for docking input)
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

try:
    from meeko import MoleculePreparation
    from meeko import PDBQTMolecule
    from meeko import RDKitMolCreate
    from meeko import PDBQTWriterLegacy
except ImportError:
    MoleculePreparation = None
    PDBQTMolecule = None
    RDKitMolCreate = None
    PDBQTWriterLegacy = None



def mol_to_compact_canonical_smiles(mol: Chem.Mol) -> Optional[str]:
    """Return compact canonical SMILES (no stereo) from a mol.
    Copies mol to a fresh RWMol (atoms + bonds only) so SDF-derived mols
    don't keep bracket-heavy SMILES (e.g. [C][N][C]...).
    """
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


def smiles_to_canonical(smiles: str) -> Optional[str]:
    """Convert SMILES string to canonical SMILES."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol_to_compact_canonical_smiles(mol)
    except Exception:
        return None

def write_prepared_ligand_sdf(smiles: str, output_sdf: str) -> bool:
    """Write single-molecule 3D SDF with explicit H (AddHs before embed/MMFF)."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return False
    canon = mol_to_compact_canonical_smiles(mol)
    if not canon:
        return False
    mol = Chem.MolFromSmiles(canon)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    w = Chem.SDWriter(output_sdf)
    w.write(mol)
    w.close()
    return Path(output_sdf).exists() and Path(output_sdf).stat().st_size > 0

def calculate_docking_box_from_sdf(sdf_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Calculate docking box from reference ligand SDF file.
    
    Returns:
        (center, size) where:
        - center: (x, y, z) coordinates of box center
        - size: (size_x, size_y, size_z) dimensions of box
    """
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Could not read SDF file: {sdf_path}")
    
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    # Calculate size: max(24, (max - min) + 5) for each dimension
    size = np.maximum(24.0, (max_coords - min_coords) + 5.0)
    
    # Center is midpoint
    center = (min_coords + max_coords) / 2.0
    
    return tuple(center), tuple(size)


def prepare_ligand_from_smiles(smiles: str, output_pdbqt: str) -> bool:
    """
    Prepare ligand from SMILES using meeko.
    
    Args:
        smiles: SMILES string
        output_pdbqt: Path to output PDBQT file
        
    Returns:
        True if successful, False otherwise
    """
    if MoleculePreparation is None:
        raise ImportError("meeko is not installed")
    
    try:
        # Convert SMILES to canonical first
        canonical_smiles = smiles_to_canonical(smiles)
        if canonical_smiles is None:
            return False
        
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return False
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        preparator = MoleculePreparation(rigid_macrocycles=True)
        setups = preparator.prepare(mol)
        if not setups:
            return False
        pdbqt_str = PDBQTWriterLegacy.write_string(setups[0])[0]
        if not pdbqt_str:
            return False
        with open(output_pdbqt, "w") as f:
            f.write(pdbqt_str)
        return Path(output_pdbqt).exists() and Path(output_pdbqt).stat().st_size > 0
    except Exception as e:
        print(f"Error preparing ligand: {e}")
        return False


def prepare_ligand_from_sdf(sdf_path: str, output_pdbqt: str) -> bool:
    """
    Prepare ligand from SDF using meeko.
    
    Args:
        sdf_path: Path to input SDF file
        output_pdbqt: Path to output PDBQT file
        
    Returns:
        True if successful, False otherwise
    """
    if MoleculePreparation is None:
        raise ImportError("meeko is not installed")
    
    try:
        mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
        if mol is None:
            return False
        
        preparator = MoleculePreparation(rigid_macrocycles=True)
        setups = preparator.prepare(mol)
        if not setups:
            return False
        pdbqt_str = PDBQTWriterLegacy.write_string(setups[0])[0]
        if not pdbqt_str:
            return False
        with open(output_pdbqt, "w") as f:
            f.write(pdbqt_str)
        return Path(output_pdbqt).exists() and Path(output_pdbqt).stat().st_size > 0
    except Exception as e:
        print(f"Error preparing ligand from SDF: {e}")
        return False


def prepare_receptor_pdb(pdb_path: str, output_pdbqt: str) -> bool:
    """
    Prepare receptor PDB file for docking.
    Uses prepare_receptor from AutoDockTools or similar.
    
    Args:
        pdb_path: Path to input PDB file
        output_pdbqt: Path to output PDBQT file
        
    Returns:
        True if successful, False otherwise
    """
    # Try using prepare_receptor from unidock tools or meeko
    try:
        from meeko import PDBQTWriterLegacy
        # For now, we'll use a simple approach - meeko can handle some PDB prep
        # But for full receptor prep, we might need AutoDockTools
        # For UniDock, we can often use PDB directly, but PDBQT is preferred
        
        # Check if unidock has a receptor preparation tool
        import subprocess
        result = subprocess.run(
            ["prepare_receptor", "-r", pdb_path, "-o", output_pdbqt],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    # If prepare_receptor not found, try using meeko's receptor preparation
    # For now, we'll assume UniDock can handle PDB files directly
    # and create a symlink or copy
    try:
        import shutil
        # UniDock can often work with PDB files directly
        # So we'll just copy it
        shutil.copy(pdb_path, output_pdbqt.replace('.pdbqt', '.pdb'))
        return True
    except Exception as e:
        print(f"Warning: Could not prepare receptor: {e}")
        print("UniDock may be able to use PDB directly")
        return False


def parse_affinity_from_pdbqt(pdbqt_path: str) -> Optional[float]:
    """Parse best binding affinity (kcal/mol) from PDBQT REMARK lines.
    Supports 'REMARK VINA RESULT' and 'minimized Affinity' formats."""
    best = None
    with open(pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("REMARK VINA RESULT"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        score = float(parts[3])
                        best = score if best is None else min(best, score)
                    except ValueError:
                        pass
            elif "REMARK" in line and ("Affinity" in line or "affinity" in line):
                m = re.search(r"[Aa]ffinity.*?([-+]?\d+\.?\d*)", line)
                if m:
                    try:
                        score = float(m.group(1))
                        best = score if best is None else min(best, score)
                    except ValueError:
                        pass
    return best


def convert_pdbqt_to_sdf(pdbqt_path: str, output_sdf: str, mol_template: Optional[Chem.Mol] = None) -> bool:
    """
    Convert PDBQT to SDF using meeko (PDBQTMolecule + RDKitMolCreate.write_sd_string).
    Preserves free_energy from Vina/UniDock REMARK lines in the SDF meeko property.
    """
    if PDBQTMolecule is None or RDKitMolCreate is None:
        return False
    try:
        pdbqt_mol = PDBQTMolecule.from_file(str(pdbqt_path), skip_typing=True)
        sdstring, failures = RDKitMolCreate.write_sd_string(pdbqt_mol)
        if not sdstring:
            return False
        with open(output_sdf, "w") as f:
            f.write(sdstring)
        return True
    except Exception as e:
        print(f"Error converting PDBQT to SDF: {e}")
        try:
            import subprocess
            subprocess.run(
                ["obabel", "-ipdbqt", pdbqt_path, "-osdf", "-O", output_sdf],
                capture_output=True, text=True, check=True,
            )
            return Path(output_sdf).exists()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Valid SMILES extraction (merged from unidock_benchmark_extract_valid_smiles.py)
# ---------------------------------------------------------------------------

# Lazy imports for extraction to avoid circular imports at module level
_extraction_imports_loaded = False


def _ensure_extraction_imports():
    global _extraction_imports_loaded
    if _extraction_imports_loaded:
        return
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    _extraction_imports_loaded = True


def validate_sample_for_docking(sample, use_simple_canonical: bool = False) -> Optional[str]:
    """
    Validate a GeneratedSample and return canonical SMILES if valid.
    Uses the same logic as the unified benchmark validity check.
    """
    from .geometry_benchmark_base import build_clean_frag_charged

    try:
        if sample.mol_repr_type == "smiles":
            smi = str(sample.mol_repr).strip()
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                return None
        else:
            sdf_path = Path(sample.mol_repr)
            supp = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
            mol0 = next(iter(supp), None)
            if mol0 is None:
                return None
            mol = Chem.RemoveHs(mol0)
            try:
                Chem.SanitizeMol(mol)
                mol = Chem.AddHs(mol)
            except Exception:
                return None

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        clean_frag = build_clean_frag_charged(sample.fragments_smi)
        if clean_frag is None or len(mol.GetSubstructMatches(clean_frag)) == 0:
            return None

        canon = Chem.MolToSmiles(mol, canonical=True) if use_simple_canonical else mol_to_compact_canonical_smiles(mol)
        return canon
    except Exception:
        return None


def extract_valid_smiles_from_instances(
    instances: list,
    use_simple_canonical: bool = False,
) -> List[Dict[str, str]]:
    """Extract valid SMILES from InstanceSamples list for docking input."""
    valid_smiles_data = []

    for inst in tqdm(instances, desc="Validating samples"):
        for sample in inst.samples:
            canon_smiles = validate_sample_for_docking(sample, use_simple_canonical=use_simple_canonical)
            if canon_smiles:
                valid_smiles_data.append({
                    "sample_name": inst.name,
                    "smiles": canon_smiles,
                    "sample_id": sample.sample_id,
                })

    return valid_smiles_data


def extract_valid_smiles_cli():
    """CLI entry point for extracting valid SMILES from generated samples."""
    _ensure_extraction_imports()
    from .source_delinker import load_instances as load_delinker_instances
    from .source_difflinker import load_instances as load_difflinker_instances
    from .source_linkllama import load_instances as load_linkllama_instances

    parser = argparse.ArgumentParser(description="Extract valid SMILES for docking")
    parser.add_argument("--method", choices=["delinker", "difflinker", "linkllama"], required=True)
    parser.add_argument("--dataset", choices=["hiqbind", "hiqbind_hard"], required=True)
    parser.add_argument("--csv", required=True, help="CSV file with sample mapping")
    parser.add_argument("--input-path", required=True, help="Path to generated samples")
    parser.add_argument("--output-csv", required=True, help="Output CSV file for valid SMILES")
    parser.add_argument("--max-samples", type=int, help="Max samples per instance")
    parser.add_argument("--expected-samples", type=int, default=100, help="Expected samples per instance")
    parser.add_argument("--simple-canonical", action="store_true",
                        help="Use rdkit-only canonical SMILES (skip mol_to_compact_canonical_smiles)")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    name_to_fragments: Dict[str, str] = {}
    name_to_gt_smiles: Dict[str, str] = {}
    for _, row in df.iterrows():
        name = row.get("name")
        if pd.isna(name):
            continue
        name = str(name).strip()
        frags = row.get("fragments", "")
        if pd.notna(frags):
            name_to_fragments[name] = str(frags).strip()
        if "SMILES" in row and pd.notna(row.get("SMILES")):
            name_to_gt_smiles[name] = str(row["SMILES"]).strip()

    input_path = Path(args.input_path)
    if args.method == "delinker":
        instances = load_delinker_instances(
            input_path=input_path,
            name_to_fragments=name_to_fragments,
            name_to_gt_smiles=name_to_gt_smiles,
            expected_samples_per_instance=args.expected_samples,
            max_samples_per_instance=args.max_samples,
        )
    elif args.method == "difflinker":
        instances = load_difflinker_instances(
            input_dir=input_path,
            name_to_fragments=name_to_fragments,
            name_to_gt_smiles=name_to_gt_smiles,
            max_samples_per_instance=args.max_samples,
        )
    elif args.method == "linkllama":
        instances = load_linkllama_instances(
            pkl_path=input_path,
            name_to_fragments=name_to_fragments,
            name_to_gt_smiles=name_to_gt_smiles,
            max_samples_per_instance=args.max_samples,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    valid_smiles_data = extract_valid_smiles_from_instances(instances, use_simple_canonical=args.simple_canonical)

    if valid_smiles_data:
        df_out = pd.DataFrame(valid_smiles_data)
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False)
        print(f"Saved {len(valid_smiles_data)} valid SMILES to {output_path}")
        return 0
    else:
        print("No valid SMILES found")
        return 1


if __name__ == "__main__":
    sys.exit(extract_valid_smiles_cli())

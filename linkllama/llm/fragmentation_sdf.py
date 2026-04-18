"""
Fragment molecules from SDF files, preserving the input 3D pose.

This module processes SDF files (or a folder of SDF files), fragments molecules
using MMPA, and computes geometric properties and energy. Unlike fragmentize.py,
it does NOT generate conformers or run energy minimization—the final pose is
taken directly from the input SDF coordinates.
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from linkllama.utils.fragmentation import fragment_dataset
from linkllama.utils.geometry import (
    compute_distance_and_angle,
    find_fragment_linker_matches,
    extract_substructure_with_3d,
)
from linkllama.utils.conformer_generation import save_conformers_to_sdf


def compute_energy_no_minimize(mol: Chem.Mol, conf_id: int = 0) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute force-field energy for a molecule with existing conformer.
    Does NOT run energy minimization—just evaluates energy at current pose.

    Args:
        mol: RDKit molecule with 3D conformer
        conf_id: Conformer ID to evaluate

    Returns:
        Tuple of (energy in kcal/mol, force_field_name) or (None, None) if failed.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None, None
    try:
        # Try MMFF first
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
            if ff is not None:
                return ff.CalcEnergy(), 'MMFF'
    except Exception:
        pass
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
        if ff is not None:
            return ff.CalcEnergy(), 'UFF'
    except Exception:
        pass
    return None, None


def process_mol_from_sdf(
    mol: Chem.Mol,
    mol_name: str,
    linker_min: int = 3,
    fragment_min: int = 5,
    min_path_length: int = 2,
    linker_leq_frags: bool = True,
    temp_dir: Optional[str] = None,
    save_sdfs: bool = True,
) -> Tuple[List[Dict], List[str]]:
    """
    Process a single molecule loaded from SDF: fragment and compute geometry.
    Uses the existing 3D coordinates—no conformer generation or minimization.

    Args:
        mol: RDKit molecule with 3D conformer from SDF
        mol_name: Identifier for the molecule
        linker_min: Minimum heavy atoms in linker
        fragment_min: Minimum heavy atoms in fragments
        min_path_length: Minimum path length between fragments
        linker_leq_frags: Linker must be <= smallest fragment
        temp_dir: Directory for output SDF files
        save_sdfs: Whether to save conformer/fragments/linker SDFs

    Returns:
        Tuple of (list of result dicts, list of temp SDF file paths)
    """
    results = []
    temp_sdf_files = []

    if mol is None or mol.GetNumConformers() == 0:
        return results, temp_sdf_files

    smiles = Chem.MolToSmiles(mol)
    if smiles is None:
        return results, temp_sdf_files

    # Remove hydrogens for geometry (match fragmentize behavior)
    mol_no_hs = Chem.RemoveHs(mol)
    if mol_no_hs is None or mol_no_hs.GetNumConformers() == 0:
        return results, temp_sdf_files

    # Fragment using existing logic
    fragmentations = fragment_dataset(
        [smiles],
        linker_min=linker_min,
        fragment_min=fragment_min,
        min_path_length=min_path_length,
        linker_leq_frags=linker_leq_frags,
        verbose=False,
    )

    if not fragmentations:
        return results, temp_sdf_files

    # Energy at current pose (no minimization)
    energy, ff_name = compute_energy_no_minimize(mol, conf_id=0)

    # Process each fragmentation
    for frag_idx, frag in enumerate(fragmentations):
        core = frag[1]  # linker (with dummy atoms)
        chains = '.'.join(frag[2:]) if len(frag) > 2 else ''  # fragments

        if not core or not chains:
            continue

        distance, angle = compute_distance_and_angle(mol_no_hs, core, chains)
        if distance is None or angle is None:
            continue

        frag_match, linker_match = find_fragment_linker_matches(mol_no_hs, core, chains)
        frags_3d_list = []
        linker_3d = None

        if frag_match:
            frags_3d = extract_substructure_with_3d(mol_no_hs, chains, list(frag_match))
            if frags_3d is not None:
                frag_components_3d = Chem.rdmolops.GetMolFrags(frags_3d, asMols=True)
                frags_3d_list = list(frag_components_3d)

        if linker_match:
            linker_3d = extract_substructure_with_3d(mol_no_hs, core, list(linker_match))

        if not frags_3d_list and linker_3d is None:
            continue

        result_dict = {
            'mol_name': mol_name,
            'smiles': smiles,
            'linker': core,
            'fragments': chains,
            'distance_angstrom': float(distance),
            'angle_radians': float(angle),
            'angle_degrees': float(angle * 180.0 / 3.141592653589793),
            'total_energy': float(energy) if energy is not None else None,
            'ff': ff_name if ff_name else None,
        }

        if save_sdfs and temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in mol_name)
            conformer_sdf = os.path.join(temp_dir, f"conformer_{safe_name}_{frag_idx}.sdf")
            save_conformers_to_sdf([mol], conformer_sdf)
            temp_sdf_files.append(conformer_sdf)
            result_dict['conformer_sdf'] = conformer_sdf

            if frags_3d_list:
                frag_sdf = os.path.join(temp_dir, f"fragments_{safe_name}_{frag_idx}.sdf")
                save_conformers_to_sdf(frags_3d_list, frag_sdf)
                temp_sdf_files.append(frag_sdf)
                result_dict['fragments_sdf'] = frag_sdf

            if linker_3d is not None:
                linker_sdf = os.path.join(temp_dir, f"linker_{safe_name}_{frag_idx}.sdf")
                save_conformers_to_sdf([linker_3d], linker_sdf)
                temp_sdf_files.append(linker_sdf)
                result_dict['linker_sdf'] = linker_sdf

        results.append(result_dict)

    return results, temp_sdf_files


def collect_sdf_paths(input_path: str) -> List[str]:
    """Collect SDF file paths from a single file or directory."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() in ('.sdf', '.sd', '.mol'):
            return [str(path)]
        return []
    if path.is_dir():
        return sorted([
            str(p) for p in path.iterdir()
            if p.suffix.lower() in ('.sdf', '.sd', '.mol')
        ])
    return []


def process_sdf_input(
    input_path: str,
    output_csv: str,
    temp_dir: Optional[str] = None,
    save_sdfs: bool = True,
    linker_min: int = 3,
    fragment_min: int = 5,
    min_path_length: int = 2,
    linker_leq_frags: bool = True,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Process SDF file(s) and write results to CSV.

    Args:
        input_path: Path to single SDF file or directory of SDF files
        output_csv: Output CSV path
        temp_dir: Directory for SDF outputs (default: same dir as output_csv)
        save_sdfs: Whether to save conformer/fragments/linker SDFs
        linker_min: Minimum heavy atoms in linker
        fragment_min: Minimum heavy atoms in fragments
        min_path_length: Minimum path length between fragments
        linker_leq_frags: Linker must be <= smallest fragment
        verbose: Print progress

    Returns:
        Tuple of (num_molecules_processed, total_fragmentations)
    """
    sdf_paths = collect_sdf_paths(input_path)
    if not sdf_paths:
        raise ValueError(f"No SDF files found at: {input_path}")

    if temp_dir is None:
        temp_dir = str(Path(output_csv).parent / "fragmentation_sdfs")
    os.makedirs(temp_dir, exist_ok=True)

    all_results = []
    mols_processed = 0

    for sdf_path in tqdm(sdf_paths, desc="Processing SDFs", disable=not verbose):
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        for mol in supplier:
            if mol is None:
                continue
            mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else Path(sdf_path).stem
            if not mol.GetNumConformers():
                if verbose:
                    tqdm.write(f"Skipping {mol_name}: no 3D coordinates")
                continue

            results, _ = process_mol_from_sdf(
                mol=mol,
                mol_name=mol_name,
                linker_min=linker_min,
                fragment_min=fragment_min,
                min_path_length=min_path_length,
                linker_leq_frags=linker_leq_frags,
                temp_dir=temp_dir if save_sdfs else None,
                save_sdfs=save_sdfs,
            )
            all_results.extend(results)
            if results:
                mols_processed += 1

    if not all_results:
        cols = ['mol_name', 'smiles', 'linker', 'fragments', 'distance_angstrom',
                'angle_radians', 'angle_degrees', 'total_energy', 'ff']
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)
        return mols_processed, 0

    df = pd.DataFrame(all_results)
    if not save_sdfs:
        for col in ('conformer_sdf', 'fragments_sdf', 'linker_sdf'):
            if col in df.columns:
                df = df.drop(columns=[col])
    else:
        # Optionally drop SDF columns for CSV (user may want them)
        # Keeping them by default; use --no-save-sdfs to avoid
        pass

    col_order = ['mol_name', 'smiles', 'linker', 'fragments', 'distance_angstrom',
                 'angle_radians', 'angle_degrees', 'total_energy', 'ff']
    if save_sdfs:
        col_order.extend([c for c in ['conformer_sdf', 'fragments_sdf', 'linker_sdf'] if c in df.columns])
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(output_csv, index=False)

    return mols_processed, len(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="Fragment molecules from SDF files, preserving input 3D pose. "
                    "Computes distance, angle, and energy (no minimization)."
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input: path to SDF file or directory of SDF files',
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        required=True,
        help='Output CSV file path',
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default=None,
        help='Directory for output SDF files (default: <output_csv_dir>/fragmentation_sdfs)',
    )
    parser.add_argument(
        '--no-save-sdfs',
        action='store_true',
        help='Do not save conformer/fragments/linker SDF files',
    )
    parser.add_argument(
        '--linker-min',
        type=int,
        default=3,
        help='Minimum heavy atoms in linker (default: 3)',
    )
    parser.add_argument(
        '--fragment-min',
        type=int,
        default=5,
        help='Minimum heavy atoms in fragments (default: 5)',
    )
    parser.add_argument(
        '--min-path-length',
        type=int,
        default=2,
        help='Minimum path length between fragments (default: 2)',
    )
    parser.add_argument(
        '--linker-leq-frags',
        action='store_true',
        default=False,
        help='Enable linker <= fragment constraint (default: disabled)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress',
    )

    args = parser.parse_args()

    mols, frags = process_sdf_input(
        input_path=args.input,
        output_csv=args.output_csv,
        temp_dir=args.temp_dir,
        save_sdfs=not args.no_save_sdfs,
        linker_min=args.linker_min,
        fragment_min=args.fragment_min,
        min_path_length=args.min_path_length,
        linker_leq_frags=args.linker_leq_frags,
        verbose=args.verbose,
    )

    print(f"Molecules processed: {mols}")
    print(f"Total fragmentations: {frags}")
    print(f"Output: {args.output_csv}")


if __name__ == '__main__':
    main()

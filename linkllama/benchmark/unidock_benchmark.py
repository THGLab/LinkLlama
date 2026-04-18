#!/usr/bin/env python3
"""
UniDock benchmarking script for HiQBind datasets.

Docks generated ligands from delinker, difflinker, and linkllama methods
against protein targets using UniDock, with support for both single-ligand
and batch (multi-ligand) docking modes.

Workflow:
  1. Extract valid SMILES (run unidock_benchmark_utils.py or use --valid-smiles-csv)
  2. Run this script to dock each valid ligand against its protein target
  3. Per-sample and master docking summary CSVs are produced automatically
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from .unidock_benchmark_utils import (
    calculate_docking_box_from_sdf,
    convert_pdbqt_to_sdf,
    parse_affinity_from_pdbqt,
    prepare_ligand_from_smiles,
)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_sample_mapping(csv_path: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """
    Load mapping from sample name to folder_name, fragments, and GT SMILES.
    Returns: (name_to_folder, name_to_fragments, name_to_gt_smiles)
    """
    df = pd.read_csv(csv_path)
    name_to_folder: Dict[str, str] = {}
    name_to_fragments: Dict[str, str] = {}
    name_to_gt_smiles: Dict[str, str] = {}

    for _, row in df.iterrows():
        name = row["name"]
        name_to_folder[name] = row["folder_name"]
        if "fragments" in row:
            name_to_fragments[name] = row["fragments"]
        if "SMILES" in row:
            name_to_gt_smiles[name] = row["SMILES"]

    return name_to_folder, name_to_fragments, name_to_gt_smiles


def load_valid_smiles_from_csv(csv_path: str, sample_name: str) -> List[Tuple[str, str]]:
    """Load valid SMILES for a single sample from a pre-extracted CSV."""
    df = pd.read_csv(csv_path)
    sample_df = df[df["sample_name"] == sample_name]
    return [(row["smiles"], row["sample_id"]) for _, row in sample_df.iterrows()]


def load_all_valid_smiles_from_csv(csv_path: str) -> Dict[str, List[Tuple[str, str]]]:
    """Load all valid SMILES grouped by sample_name from a pre-extracted CSV."""
    df = pd.read_csv(csv_path)
    out: Dict[str, List[Tuple[str, str]]] = {}
    if df.empty:
        return out
    required = {"sample_name", "smiles", "sample_id"}
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"valid_smiles CSV missing columns: {sorted(missing)}")
    for sample_name, g in df.groupby("sample_name", sort=False):
        out[str(sample_name)] = [(r["smiles"], r["sample_id"]) for _, r in g.iterrows()]
    return out


# ---------------------------------------------------------------------------
# UniDock invocation
# ---------------------------------------------------------------------------

def run_unidock_single(
    receptor: str,
    ligand_pdbqt: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    output_dir: str,
    search_mode: str = "fast",
) -> bool:
    """Run UniDock for a single ligand. Returns True on success."""
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    output_pdbqt = output_dir_p / "docked_poses.pdbqt"

    try:
        cmd = [
            "unidock",
            "--receptor", receptor,
            "--ligand", ligand_pdbqt,
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
            "--search_mode", search_mode,
            "--out", str(output_pdbqt),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return output_pdbqt.exists()
    except subprocess.CalledProcessError as e:
        print(f"UniDock failed: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr[:500]}")
        return False


def run_unidock_multi(
    receptor: str,
    ligand_index_file: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    output_dir: str,
    search_mode: str = "fast",
) -> bool:
    """Run UniDock in multi-ligand mode via --ligand_index. Returns True on success."""
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "unidock",
            "--receptor", receptor,
            "--ligand_index", ligand_index_file,
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(size[0]),
            "--size_y", str(size[1]),
            "--size_z", str(size[2]),
            "--search_mode", search_mode,
            "--dir", str(output_dir_p),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return any(output_dir_p.glob("*_out.pdbqt"))
    except subprocess.CalledProcessError as e:
        print(f"UniDock (multi) failed: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr[:500]}")
        return False


# ---------------------------------------------------------------------------
# Post-docking SDF extraction
# ---------------------------------------------------------------------------

def _extract_best_pose(
    out_pdbqt: Path,
    output_dir: Path,
    sample_name: str,
    sample_id: str,
    smiles: str,
    method: str,
) -> Optional[dict]:
    """
    Convert the best pose from a docked PDBQT to SDF and return a summary dict,
    or None on failure.
    """
    if not out_pdbqt.is_file():
        return None

    temp_sdf = out_pdbqt.with_suffix(".sdf")
    if not convert_pdbqt_to_sdf(str(out_pdbqt), str(temp_sdf)):
        print(f"Warning: Could not convert PDBQT to SDF for {sample_name} {sample_id}")
        return None

    best_energy = parse_affinity_from_pdbqt(str(out_pdbqt))
    suppl = Chem.SDMolSupplier(str(temp_sdf), removeHs=False)
    for mol in suppl:
        if mol is None:
            continue
        output_sdf = output_dir / f"{sample_id}_{method}_0.sdf"
        writer = Chem.SDWriter(str(output_sdf))
        mol.SetProp("_Name", f"{sample_id}_{method}_0")
        mol.SetProp("_SampleName", sample_name)
        mol.SetProp("_SampleID", sample_id)
        mol.SetProp("_Method", method)
        mol.SetProp("_PoseIndex", "0")
        mol.SetProp("_SMILES", smiles)
        writer.write(mol)
        writer.close()
        return {
            "sample_name": sample_name,
            "sample_id": sample_id,
            "method": method,
            "pose_index": 0,
            "sdf_file": output_sdf.name,
            "sdf_path": str(output_sdf),
            "score": best_energy,
        }
    return None


# ---------------------------------------------------------------------------
# Receptor preparation
# ---------------------------------------------------------------------------

def _prepare_receptor(
    protein_pdb: Path,
    output_dir: Path,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
) -> Optional[Path]:
    """
    Clean a PDB and run mk_prepare_receptor.py to produce a PDBQT.
    Returns the path to the receptor PDBQT, or None on failure.
    """
    cleaned_pdb = output_dir / "receptor.pdb"
    with open(protein_pdb, "r") as fin, open(cleaned_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM", "TER", "END", "ENDMDL", "MODEL")):
                fout.write(line)

    receptor_pdbqt = output_dir / "receptor.pdbqt"
    prep_cmd = [
        "mk_prepare_receptor.py",
        "--read_pdb", str(cleaned_pdb.name),
        "-o", "receptor",
        "-p", "-v",
        "--box_size", str(size[0]), str(size[1]), str(size[2]),
        "--box_center", str(center[0]), str(center[1]), str(center[2]),
    ]
    try:
        subprocess.run(prep_cmd, cwd=str(output_dir), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("mk_prepare_receptor.py failed:")
        if e.stderr:
            print(e.stderr[:500])
        return None

    if not receptor_pdbqt.exists():
        alt = output_dir / "receptor_rigid.pdbqt"
        if alt.exists():
            return alt
        print("mk_prepare_receptor.py did not produce receptor.pdbqt")
        return None

    return receptor_pdbqt


# ---------------------------------------------------------------------------
# Core docking logic for one sample
# ---------------------------------------------------------------------------

def dock_single_sample(
    sample_name: str,
    folder_name: str,
    method: str,
    dataset: str,
    base_data_dir: str,
    base_output_dir: str,
    generated_samples_base: str,
    name_to_fragments: Dict[str, str],
    name_to_gt_smiles: Dict[str, str],
    valid_smiles_csv: Optional[str] = None,
    valid_smiles_list: Optional[List[Tuple[str, str]]] = None,
    max_ligands_per_sample: Optional[int] = None,
    ligand_pbar: Optional[tqdm] = None,
    ligand_stats: Optional[Dict[str, int]] = None,
    multi_ligand: bool = False,
) -> bool:
    """Dock all valid ligands for one sample. Returns True on success."""

    # --- Resolve reference paths ---
    data_dir = Path(base_data_dir) / f"1k_{dataset}" / "sdfs" / "raw" / folder_name
    ref_ligand_sdf = data_dir / f"{folder_name}_ligand_refined.sdf"
    protein_pdb = data_dir / f"{folder_name}_protein_refined.pdb"

    if not ref_ligand_sdf.exists() or not protein_pdb.exists():
        print(f"Missing reference files for {sample_name}")
        return False

    # --- Calculate docking box ---
    try:
        center, size = calculate_docking_box_from_sdf(str(ref_ligand_sdf))
    except Exception as e:
        print(f"Error calculating box for {sample_name}: {e}")
        return False

    # --- Gather valid SMILES ---
    if valid_smiles_list is not None:
        valid_smiles = valid_smiles_list
    elif valid_smiles_csv and Path(valid_smiles_csv).exists():
        valid_smiles = load_valid_smiles_from_csv(valid_smiles_csv, sample_name)
    else:
        print(f"Warning: No valid SMILES CSV provided for {sample_name}")
        return False

    if not valid_smiles:
        print(f"No valid SMILES found for {sample_name} ({method})")
        return False

    # --- Set up output directory & skip if already done ---
    output_dir = Path(base_output_dir) / dataset / method / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "docking_summary.csv"
    if summary_csv.is_file():
        n_skip = len(valid_smiles) if max_ligands_per_sample is None else min(len(valid_smiles), max_ligands_per_sample)
        if ligand_pbar is not None:
            ligand_pbar.update(n_skip)
        print(f"Skipping {sample_name} (already done: {summary_csv})")
        return True

    # --- Prepare receptor ---
    receptor_pdbqt = _prepare_receptor(protein_pdb, output_dir, center, size)
    if receptor_pdbqt is None:
        return False

    # --- Dock ligands ---
    temp_dir = Path(tempfile.mkdtemp(prefix=f"dock_{sample_name}_"))
    ligands_to_dock = valid_smiles
    if max_ligands_per_sample is not None:
        ligands_to_dock = ligands_to_dock[:max_ligands_per_sample]

    print(f"Docking {len(ligands_to_dock)} ligands for {sample_name}...")

    try:
        if multi_ligand:
            docked_sdfs = _dock_multi_ligand(
                ligands_to_dock, sample_name, method, receptor_pdbqt,
                center, size, output_dir, temp_dir, ligand_pbar, ligand_stats,
            )
        else:
            docked_sdfs = _dock_single_ligand_loop(
                ligands_to_dock, sample_name, method, receptor_pdbqt,
                center, size, output_dir, temp_dir, ligand_pbar, ligand_stats,
            )

        # --- Save outputs ---
        _save_docking_config(output_dir, sample_name, method, dataset, center, size,
                            ref_ligand_sdf, protein_pdb, receptor_pdbqt)
        _save_docking_summary(output_dir, summary_csv, docked_sdfs, sample_name, method, center, size)
        return bool(docked_sdfs)

    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")


# ---------------------------------------------------------------------------
# Multi-ligand docking
# ---------------------------------------------------------------------------

def _dock_multi_ligand(
    ligands_to_dock: List[Tuple[str, str]],
    sample_name: str,
    method: str,
    receptor_pdbqt: Path,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    output_dir: Path,
    temp_dir: Path,
    ligand_pbar: Optional[tqdm],
    ligand_stats: Optional[Dict[str, int]],
) -> List[dict]:
    """Prepare all ligands, dock in one UniDock call, extract poses."""
    docked_sdfs: List[dict] = []
    ligand_input_dir = temp_dir / "input"
    ligand_input_dir.mkdir(parents=True, exist_ok=True)

    prepared_entries: List[Tuple[str, str]] = []
    for idx, (smiles, sample_id) in enumerate(ligands_to_dock):
        prep_index = len(prepared_entries)
        ligand_pdbqt = ligand_input_dir / f"{prep_index}.pdbqt"
        if not prepare_ligand_from_smiles(smiles, str(ligand_pdbqt)):
            print(f"Failed to prepare ligand {idx} ({sample_id}) for {sample_name}")
            if ligand_pbar is not None:
                ligand_pbar.update(1)
            continue
        prepared_entries.append((sample_id, smiles))

    if not prepared_entries:
        print(f"No ligands successfully prepared for {sample_name}")
        return docked_sdfs

    ligand_index_file = temp_dir / "ligands.txt"
    with open(ligand_index_file, "w") as f:
        f.write(" ".join(str(ligand_input_dir / f"{i}.pdbqt") for i in range(len(prepared_entries))))

    dock_output_dir = temp_dir / "dock_batch"
    dock_output_dir.mkdir(parents=True, exist_ok=True)

    if not run_unidock_multi(str(receptor_pdbqt), str(ligand_index_file), center, size, str(dock_output_dir)):
        print(f"Multi-ligand UniDock failed for {sample_name}")
        return docked_sdfs

    for i, (sample_id, smiles) in enumerate(prepared_entries):
        out_pdbqt = dock_output_dir / f"{i}_out.pdbqt"
        result = _extract_best_pose(out_pdbqt, output_dir, sample_name, sample_id, smiles, method)
        if result is not None:
            docked_sdfs.append(result)
            print(f"Successfully docked ligand (multi) index {i} ({sample_id}) - 1 pose")
            _update_ligand_progress(ligand_pbar, ligand_stats, docked=True)
        else:
            _update_ligand_progress(ligand_pbar, ligand_stats, docked=False)

    return docked_sdfs


# ---------------------------------------------------------------------------
# Single-ligand docking loop
# ---------------------------------------------------------------------------

def _dock_single_ligand_loop(
    ligands_to_dock: List[Tuple[str, str]],
    sample_name: str,
    method: str,
    receptor_pdbqt: Path,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    output_dir: Path,
    temp_dir: Path,
    ligand_pbar: Optional[tqdm],
    ligand_stats: Optional[Dict[str, int]],
) -> List[dict]:
    """Dock each ligand individually. Returns list of docking result dicts."""
    docked_sdfs: List[dict] = []

    for idx, (smiles, sample_id) in enumerate(ligands_to_dock):
        ligand_pdbqt = temp_dir / f"ligand_{idx}_{sample_id}.pdbqt"
        if not prepare_ligand_from_smiles(smiles, str(ligand_pdbqt)):
            print(f"Failed to prepare ligand {idx} ({sample_id}) for {sample_name}")
            _update_ligand_progress(ligand_pbar, ligand_stats, docked=False)
            continue

        dock_output = temp_dir / f"dock_{idx}_{sample_id}"
        dock_output.mkdir(parents=True, exist_ok=True)

        if not run_unidock_single(str(receptor_pdbqt), str(ligand_pdbqt), center, size, str(dock_output)):
            print(f"Failed to dock ligand {idx} ({sample_id}) for {sample_name}")
            _update_ligand_progress(ligand_pbar, ligand_stats, docked=False)
            continue

        docked_pdbqt = dock_output / "docked_poses.pdbqt"
        result = _extract_best_pose(docked_pdbqt, output_dir, sample_name, sample_id, smiles, method)
        if result is not None:
            docked_sdfs.append(result)
            print(f"Successfully docked ligand {idx} ({sample_id}) - 1 pose")

        _update_ligand_progress(ligand_pbar, ligand_stats, docked=(result is not None))

    return docked_sdfs


# ---------------------------------------------------------------------------
# Progress & I/O helpers
# ---------------------------------------------------------------------------

def _update_ligand_progress(
    ligand_pbar: Optional[tqdm],
    ligand_stats: Optional[Dict[str, int]],
    docked: bool,
) -> None:
    if ligand_pbar is not None:
        ligand_pbar.update(1)
    if docked and ligand_stats is not None:
        ligand_stats["docked_ok"] = ligand_stats.get("docked_ok", 0) + 1
    if ligand_pbar is not None and ligand_stats is not None:
        ligand_pbar.set_postfix(docked_ok=ligand_stats.get("docked_ok", 0), refresh=False)


def _save_docking_config(
    output_dir: Path,
    sample_name: str,
    method: str,
    dataset: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    ref_ligand_sdf: Path,
    protein_pdb: Path,
    receptor_pdbqt: Path,
) -> None:
    config_file = output_dir / "docking_config.txt"
    with open(config_file, "w") as f:
        f.write(f"Sample: {sample_name}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Box center: {center}\n")
        f.write(f"Box size: {size}\n")
        f.write(f"Search mode: fast\n")
        f.write(f"Reference ligand: {ref_ligand_sdf}\n")
        f.write(f"Protein PDB: {protein_pdb}\n")
        f.write(f"Receptor PDBQT: {receptor_pdbqt}\n")


def _save_docking_summary(
    output_dir: Path,
    summary_csv: Path,
    docked_sdfs: List[dict],
    sample_name: str,
    method: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
) -> None:
    if docked_sdfs:
        df_summary = pd.DataFrame(docked_sdfs)
        df_summary.to_csv(summary_csv, index=False)

        n_ligands = len({row["sample_id"] for row in docked_sdfs})
        overall = pd.DataFrame({
            "sample_name": [sample_name],
            "method": [method],
            "n_ligands_docked": [n_ligands],
            "total_poses": [len(docked_sdfs)],
            "box_center": [str(center)],
            "box_size": [str(size)],
        })
        overall.to_csv(output_dir / "docking_overall_summary.csv", index=False)

        print(f"Successfully docked {n_ligands} ligands ({len(docked_sdfs)} poses) for {sample_name}")
        print(f"Saved summary to {summary_csv}")
    else:
        print(f"No successful docks for {sample_name}")
        pd.DataFrame([{
            "sample_name": sample_name,
            "method": method,
            "n_ligands_docked": 0,
            "total_poses": 0,
            "error": "No successful docks",
        }]).to_csv(summary_csv, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UniDock benchmarking for HiQBind datasets")
    parser.add_argument("--dataset", choices=["hiqbind", "hiqbind_hard"], required=True)
    parser.add_argument("--method", choices=["delinker", "difflinker", "linkllama"], required=True)
    parser.add_argument("--sample-name", help="Single sample to process (for testing)")
    parser.add_argument("--max-ligands-per-sample", type=int, default=None,
                        help="Max ligands to dock per sample (default: all)")
    parser.add_argument("--multi-ligand-per-sample", action="store_true",
                        help="Use UniDock multi-ligand mode (one call per sample via --ligand_index)")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--output-dir", default="benchmark_unidock", help="Output directory")
    parser.add_argument("--generated-samples-base", default="baseline_results",
                        help="Base directory for generated samples")
    parser.add_argument("--csv",
                        help="CSV file with sample mapping (default: data/1k_{dataset}/hiqbind_1k_...csv)")
    parser.add_argument("--valid-smiles-csv",
                        help="CSV file with pre-extracted valid SMILES (for docking)")

    args = parser.parse_args()

    # --- Determine CSV path ---
    if args.csv:
        csv_path = args.csv
    else:
        dataset_suffix = args.dataset.replace("hiqbind", "random").replace("_hard", "_hard")
        csv_path = str(
            Path(args.data_dir) / f"1k_{args.dataset}"
            / f"hiqbind_1k_{dataset_suffix}_with_reasonability.csv"
        )

    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return 1

    name_to_folder, name_to_fragments, name_to_gt_smiles = load_sample_mapping(csv_path)

    # --- Resolve valid SMILES CSV ---
    valid_smiles_csv = args.valid_smiles_csv
    if not valid_smiles_csv:
        valid_smiles_csv = str(
            Path(args.output_dir) / "valid_smiles" / f"{args.method}_{args.dataset}_valid_smiles.csv"
        )

    # --- Single-sample mode ---
    if args.sample_name:
        if args.sample_name not in name_to_folder:
            print(f"Sample {args.sample_name} not found in mapping")
            return 1

        success = dock_single_sample(
            args.sample_name,
            name_to_folder[args.sample_name],
            args.method, args.dataset,
            args.data_dir, args.output_dir, args.generated_samples_base,
            name_to_fragments, name_to_gt_smiles,
            valid_smiles_csv,
            max_ligands_per_sample=args.max_ligands_per_sample,
            multi_ligand=args.multi_ligand_per_sample,
        )
        return 0 if success else 1

    # --- Batch mode ---
    print(f"Processing all samples for {args.method} on {args.dataset}")

    valid_smiles_by_sample: Dict[str, List[Tuple[str, str]]] = {}
    if valid_smiles_csv and Path(valid_smiles_csv).exists():
        valid_smiles_by_sample = load_all_valid_smiles_from_csv(valid_smiles_csv)

    total_ligands = 0
    for sample_name in name_to_folder:
        ligs = valid_smiles_by_sample.get(sample_name, [])
        if args.max_ligands_per_sample is not None:
            ligs = ligs[: args.max_ligands_per_sample]
        total_ligands += len(ligs)

    ligand_stats: Dict[str, int] = {"docked_ok": 0}
    ligand_pbar = tqdm(total=total_ligands, desc="Ligands docked", unit="lig")
    sample_pbar = tqdm(name_to_folder.items(), desc="Samples", unit="sample")

    success_count = 0
    total_count = 0
    for sample_name, folder_name in sample_pbar:
        total_count += 1
        if dock_single_sample(
            sample_name, folder_name,
            args.method, args.dataset,
            args.data_dir, args.output_dir, args.generated_samples_base,
            name_to_fragments, name_to_gt_smiles,
            valid_smiles_csv,
            valid_smiles_list=valid_smiles_by_sample.get(sample_name),
            max_ligands_per_sample=args.max_ligands_per_sample,
            ligand_pbar=ligand_pbar,
            ligand_stats=ligand_stats,
            multi_ligand=args.multi_ligand_per_sample,
        ):
            success_count += 1

    ligand_pbar.close()
    print(f"Successfully docked {success_count}/{total_count} samples")

    # --- Build master CSV ---
    master_rows = []
    base_dir = Path(args.output_dir) / args.dataset / args.method
    for sample_dir in base_dir.glob("*"):
        summary_path = sample_dir / "docking_summary.csv"
        if summary_path.is_file():
            try:
                master_rows.append(pd.read_csv(summary_path))
            except Exception:
                continue
    if master_rows:
        master_out = Path(args.output_dir) / "docking_results" / f"{args.method}_{args.dataset}_docking.csv"
        master_out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(master_rows, ignore_index=True).to_csv(master_out, index=False)
        print(f"Wrote master docking CSV to {master_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

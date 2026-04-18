#!/usr/bin/env python3
"""
Standalone script to calculate all molecular properties from sft_corpus.py
and append them as new columns to chembl36_cleaned_frags.csv.

This script processes the CSV in chunks to handle large files efficiently.
"""

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Optional, Dict, Any

# Add parent directory to path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from linkllama.utils.fragmentation import remove_dummys
from linkllama.utils.properties import (
    has_undesirable_pattern, 
    has_pains_alert, 
    has_bad_ring, 
    check_ring_system, 
    check_reos,
    get_linker_topology,
    get_linker_num_rotatable_bonds,
    get_linker_num_heavy_atoms,
    get_mol_mw,
    get_mol_logp,
    get_mol_tpsa,
    get_mol_num_hbd,
    get_mol_num_hba
)


def get_overall_reasonability(result: Dict[str, Any]) -> str:
    """Return 'reasonable' if all checks pass, 'unreasonable' otherwise."""
    if result.get('linker_bad_ring'):
        return "unreasonable"
    if str(result.get('linker_problematic_ring', '')).strip() not in ('', 'nan'):
        return "unreasonable"
    if result.get('mol_has_undesirable_pattern') or result.get('mol_pains_alert'):
        return "unreasonable"
    if str(result.get('mol_reos_failed_rule', '')).strip() not in ('', 'nan'):
        return "unreasonable"
    return "reasonable"


def calculate_properties_row(
    row_dict: Dict[str, Any], 
    smiles_column: str = 'SMILES', 
    linker_column: str = 'linker'
) -> Dict[str, Any]:
    """
    Calculate all properties for a single row.
    Takes a dictionary (row.to_dict()) instead of a pandas Series for multiprocessing.
    Returns a dictionary with all calculated properties.
    """
    result = {}
    
    # Parse row data
    full_smiles = row_dict[smiles_column]
    linker_smi = row_dict.get(linker_column, '')
    
    # Create full molecule
    full_mol = Chem.MolFromSmiles(full_smiles)
    result['mol_weight'] = get_mol_mw(full_mol)
    result['mol_logp'] = get_mol_logp(full_mol)
    result['mol_tpsa'] = get_mol_tpsa(full_mol)
    result['mol_num_hbd'] = get_mol_num_hbd(full_mol)
    result['mol_num_hba'] = get_mol_num_hba(full_mol)
    
    # Calculate Molecule reasonability
    result['mol_has_undesirable_pattern'] = has_undesirable_pattern(full_mol)
    result['mol_pains_alert'] = has_pains_alert(full_mol)
    result['mol_reos_failed_rule'] = check_reos(full_smiles)
    
    # Calculate linker properties
    linker_cleaned_smi = remove_dummys(linker_smi)
    linker_mol = Chem.MolFromSmiles(linker_cleaned_smi)
    result['linker_num_rotatable_bonds'] = get_linker_num_rotatable_bonds(linker_mol)
    result['linker_num_heavy_atoms'] = get_linker_num_heavy_atoms(linker_mol)
    result['linker_topology'] = get_linker_topology(linker_mol)
    
    # Calculate Linker reasonability
    result['linker_bad_ring'] = has_bad_ring(linker_mol)
    result['linker_problematic_ring'] = check_ring_system(linker_cleaned_smi)
    
    result['overall_reasonability'] = get_overall_reasonability(result)
    
    return result


def _process_row_worker(args):
    """Worker function for parallel processing - processes a single row."""
    row_dict, idx = args
    try:
        props = calculate_properties_row(row_dict)
        return (props, idx, None)
    except Exception as e:
        # Return None to indicate error - row will be dropped
        return (None, idx, e)


def process_chunk(
    chunk_df: pd.DataFrame, 
    num_workers: int = 1, 
    verbose: bool = False
) -> pd.DataFrame:
    """
    Process a chunk of the dataframe and calculate all properties.
    Returns a DataFrame with properties appended.
    Uses parallel processing if num_workers > 1.
    Drops rows that have errors during processing.
    """
    # Convert rows to dictionaries for multiprocessing
    row_dicts = []
    indices = []
    for idx, row in chunk_df.iterrows():
        row_dicts.append(row.to_dict())
        indices.append(idx)
    
    successful_results = []  # List of combined dicts
    error_count = 0
    
    if num_workers > 1 and len(row_dicts) > 0:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = [(row_dict, idx) for row_dict, idx in zip(row_dicts, indices)]
            futures = {executor.submit(_process_row_worker, task): i for i, task in enumerate(tasks)}
            
            for future in as_completed(futures):
                task_idx = futures[future]
                props, idx, error = future.result()
                if error is None and props is not None:
                    # Combine original row with properties
                    combined = row_dicts[task_idx].copy()
                    combined.update(props)
                    successful_results.append(combined)
                else:
                    error_count += 1
                    if verbose:
                        print(f"Error processing row {idx}: {error}")
    else:
        # Sequential processing
        for row_dict, idx in zip(row_dicts, indices):
            try:
                props = calculate_properties_row(row_dict)
                combined = row_dict.copy()
                combined.update(props)
                successful_results.append(combined)
            except Exception as e:
                error_count += 1
                if verbose:
                    print(f"Error processing row {idx}: {e}")
    
    if error_count > 0:
        print(f"Dropped {error_count} rows with errors from chunk")
    
    if not successful_results:
        return pd.DataFrame()
    
    return pd.DataFrame(successful_results)


def process_chunk_and_write(
    chunk_df: pd.DataFrame, 
    output_csv_path: str,
    header_written: bool,
    num_workers: int = 1, 
    verbose: bool = False
) -> tuple[bool, int, int]:
    """
    Process a chunk of the dataframe, calculate properties, and write directly to CSV.
    Uses parallel processing if num_workers > 1.
    Drops rows that have errors during processing.
    
    Returns:
        tuple: (header_written, successful_count, error_count)
    """
    # Convert rows to dictionaries for multiprocessing
    row_dicts = []
    indices = []
    for idx, row in chunk_df.iterrows():
        row_dicts.append(row.to_dict())
        indices.append(idx)
    
    successful_results = []  # List of (row_dict, props_dict) tuples
    error_count = 0
    
    if num_workers > 1 and len(row_dicts) > 0:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = [(row_dict, idx) for row_dict, idx in zip(row_dicts, indices)]
            futures = {executor.submit(_process_row_worker, task): i for i, task in enumerate(tasks)}
            
            for future in as_completed(futures):
                task_idx = futures[future]
                props, idx, error = future.result()
                if error is None and props is not None:
                    # Successful processing - combine original row with properties
                    combined = row_dicts[task_idx].copy()
                    combined.update(props)
                    successful_results.append(combined)
                else:
                    # Error occurred - drop this row
                    error_count += 1
                    if verbose:
                        print(f"Error processing row {idx}: {error}")
    else:
        # Sequential processing
        for row_dict, idx in zip(row_dicts, indices):
            try:
                props = calculate_properties_row(row_dict)
                # Combine original row with properties
                combined = row_dict.copy()
                combined.update(props)
                successful_results.append(combined)
            except Exception as e:
                # Error occurred - drop this row
                error_count += 1
                if verbose:
                    print(f"Error processing row {idx}: {e}")
    
    if error_count > 0:
        print(f"Dropped {error_count} rows with errors from chunk")
    
    # Write successful results directly to CSV
    if successful_results:
        result_df = pd.DataFrame(successful_results)
        result_df.to_csv(
            output_csv_path, 
            mode='a' if header_written else 'w',
            index=False, 
            header=not header_written
        )
        return True, len(successful_results), error_count
    
    return header_written, 0, error_count


def main(
    input_csv_path: Path,
    output_csv_path: Path,
    chunk_size: int = 12800,
    num_workers: Optional[int] = None,
    verbose: bool = False
) -> None:
    """
    Main function to process CSV in chunks and calculate all properties.
    
    Args:
        input_csv_path: Path to input CSV file
        output_csv_path: Path to output CSV file with properties appended
        chunk_size: Number of rows to process at a time (default: 12800)
        num_workers: Number of parallel workers (default: cpu_count() // 4)
        verbose: Print verbose error messages (default: False)
    """
    print(f"Reading CSV from: {input_csv_path}")
    print(f"Processing in chunks of {chunk_size} rows")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 4
    print(f"Using {num_workers} workers")
    
    # Read header to get original columns
    header_df = pd.read_csv(str(input_csv_path), nrows=0)
    original_columns = list(header_df.columns)
    original_columns_set = set(original_columns)
    
    # Get total number of rows for progress tracking
    total_rows = sum(1 for _ in open(input_csv_path)) - 1  # Subtract header
    print(f"Total rows to process: {total_rows:,}")
    
    # Process file in chunks and write incrementally
    processed = 0
    total_successful = 0
    total_errors = 0
    header_written = False
    
    pbar = tqdm(total=total_rows, desc="Processing CSV", unit="rows")
    
    for chunk_df in pd.read_csv(str(input_csv_path), chunksize=chunk_size):
        header_written, successful, errors = process_chunk_and_write(
            chunk_df, 
            str(output_csv_path),
            header_written,
            num_workers=num_workers, 
            verbose=verbose
        )
        
        total_successful += successful
        total_errors += errors
        processed += len(chunk_df)
        pbar.update(len(chunk_df))
        
        if verbose:
            print(f"Processed {processed:,} / {total_rows:,} rows ({100*processed/total_rows:.1f}%)")
    
    pbar.close()
    
    print("Done!")
    print(f"Total successful: {total_successful:,}, Total errors: {total_errors:,}")
    
    # Read final output to get stats (only if file has data)
    if total_successful > 0:
        result_df = pd.read_csv(str(output_csv_path))
        print(f"Output file contains {len(result_df)} rows and {len(result_df.columns)} columns")
        new_cols = [col for col in result_df.columns if col not in original_columns_set]
        print(f"New columns added: {new_cols}")
    else:
        print("WARNING: No rows were successfully processed!")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate molecular properties from CSV file and append as new columns. "
            "Processes the CSV in chunks to handle large files efficiently. "
            "Rows with errors during processing are dropped."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s input.csv output.csv\n"
            "  %(prog)s input.csv output.csv --chunk-size 10000\n"
            "  %(prog)s input.csv output.csv --num-workers 8 --verbose\n"
            "  %(prog)s input.csv output.csv -c 5000 -w 4 -v\n"
        )
    )
    
    parser.add_argument(
        'input_csv',
        type=Path,
        help='Path to input CSV file containing SMILES and linker columns'
    )
    
    parser.add_argument(
        'output_csv',
        type=Path,
        help='Path to output CSV file with calculated properties appended'
    )
    
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=12800,
        metavar='N',
        help='Number of rows to process per chunk (default: 12800)'
    )
    
    parser.add_argument(
        '-w', '--num-workers',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Number of parallel workers. '
            f'Default: cpu_count() // 4 ({multiprocessing.cpu_count() // 4})'
        )
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print verbose error messages for each failed row'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not args.input_csv.exists():
        parser.error(f"Input file does not exist: {args.input_csv}")
    
    if not args.input_csv.is_file():
        parser.error(f"Input path is not a file: {args.input_csv}")
    
    # Validate chunk_size
    if args.chunk_size <= 0:
        parser.error(f"chunk_size must be positive, got: {args.chunk_size}")
    
    # Validate num_workers
    if args.num_workers is not None and args.num_workers <= 0:
        parser.error(f"num_workers must be positive, got: {args.num_workers}")
    
    max_workers = multiprocessing.cpu_count()
    if args.num_workers is not None and args.num_workers > max_workers:
        parser.error(
            f"num_workers ({args.num_workers}) exceeds available CPUs ({max_workers})"
        )
    
    # Check if output directory exists, create if not
    output_dir = args.output_csv.parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        verbose=args.verbose
    )


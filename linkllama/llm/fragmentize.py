"""
Main module for fragmentizing molecules with 3D geometry information.

This module orchestrates the fragmentation and 3D geometry calculation pipeline:
1. Generate 3D conformers from SMILES strings
2. Fragment molecules into cores and linkers
3. Compute geometric properties (distance and angle)
4. Save results in a randomized way
"""

from rdkit import Chem
from typing import List, Dict, Optional, Tuple
import json
import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add parent directory to path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from linkllama.utils.fragmentation import fragment_dataset
from linkllama.utils.geometry import (
    compute_distance_and_angle,
    find_fragment_linker_matches,
    extract_substructure_with_3d
)
from linkllama.utils.conformer_generation import generate_randomized_conformer, save_conformers_to_sdf

def process_smiles_with_geometry(smiles: str,
                                  linker_min: int = 3,
                                  fragment_min: int = 5,
                                  min_path_length: int = 2,
                                  linker_leq_frags: bool = True,
                                  num_confs: int = 10,
                                  random_seed: Optional[int] = None,
                                  temp_dir: Optional[str] = None,
                                  molecule_name: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
    """Process a single SMILES string: generate conformer, fragment, and compute geometry."""
    results = []
    temp_sdf_files = []
    
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
    
    # Generate conformer
    mol, energy, force_field_name = generate_randomized_conformer(smiles, num_confs=num_confs, random_seed=random_seed)
    if mol is None or mol.GetNumConformers() == 0:
        return results, temp_sdf_files
    
    # Save conformer SDF - use molecule_name if provided, otherwise use hash
    if molecule_name is not None:
        name_suffix = molecule_name
    else:
        name_suffix = str(abs(hash(smiles)) % 1000000)
    temp_sdf = os.path.join(temp_dir or '/tmp', f"conformer_{name_suffix}.sdf")
    save_conformers_to_sdf([mol], temp_sdf)
    temp_sdf_files.append(temp_sdf)
    
    # Filter fragmentations
    fragmentations = fragment_dataset([smiles], linker_min=linker_min, fragment_min=fragment_min,
                                     min_path_length=min_path_length, linker_leq_frags=linker_leq_frags,
                                     verbose=False)
    
    if not fragmentations:
        return results, temp_sdf_files
    
    # Remove hydrogens from conformer for geometry calculation
    mol_no_hs = Chem.RemoveHs(mol)
    if mol_no_hs is None or mol_no_hs.GetNumConformers() == 0:
        return results, temp_sdf_files
    
    # Process each fragmentation
    for frag_idx, frag in enumerate(fragmentations):
        core = frag[1]  # linker (with dummy atoms)
        chains = '.'.join(frag[2:]) if len(frag) > 2 else ''  # fragments (with dummy atoms)
        
        if not core or not chains:
            continue
        
        # Compute distance and angle
        distance, angle = compute_distance_and_angle(mol_no_hs, core, chains)
        
        if distance is not None and angle is not None:
            frag_match, linker_match = find_fragment_linker_matches(mol_no_hs, core, chains)
            
            # Extract fragments and linker with 3D coordinates
            frags_3d_list = []
            linker_3d = None
            
            if frag_match:
                frags_3d = extract_substructure_with_3d(mol_no_hs, chains, list(frag_match))
                if frags_3d is not None:
                    frag_components_3d = Chem.rdmolops.GetMolFrags(frags_3d, asMols=True)
                    frags_3d_list = list(frag_components_3d)
            
            if linker_match:
                linker_3d = extract_substructure_with_3d(mol_no_hs, core, list(linker_match))
            
            if frags_3d_list or linker_3d is not None:
                frag_sdf = os.path.join(temp_dir or '/tmp', f"fragments_{frag_idx}_{name_suffix}.sdf")
                linker_sdf = os.path.join(temp_dir or '/tmp', f"linker_{frag_idx}_{name_suffix}.sdf")
                
                if frags_3d_list:
                    save_conformers_to_sdf(frags_3d_list, frag_sdf)
                    temp_sdf_files.append(frag_sdf)
                
                if linker_3d is not None:
                    save_conformers_to_sdf([linker_3d], linker_sdf)
                    temp_sdf_files.append(linker_sdf)
                
                result_dict = {
                    'smiles': smiles,
                    'linker': core,
                    'fragments': chains,
                    'distance_angstrom': float(distance),
                    'angle_radians': float(angle),
                    'angle_degrees': float(angle * 180.0 / 3.141592653589793),
                    'conformer_sdf': temp_sdf,
                    'fragments_sdf': frag_sdf if frags_3d_list else None,
                    'linker_sdf': linker_sdf if linker_3d is not None else None,
                    'total_energy': float(energy) if energy is not None else None,
                    'ff': force_field_name if force_field_name is not None else None
                }
                
                results.append(result_dict)
    
    return results, temp_sdf_files


def _save_results_to_csv(results: List[Dict], output_csv: str, original_columns: List[str] = None):
    """Convert results to DataFrame and save to CSV, removing SDF columns."""
    if not results:
        output_columns = (original_columns or ['SMILES']) + ['linker', 'fragments', 
                    'distance_angstrom', 'angle_radians', 'angle_degrees', 'total_energy', 'ff']
        pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False)
        return
    
    result_df = pd.DataFrame(results)
    
    # Remove SDF file columns
    sdf_columns = ['conformer_sdf', 'fragments_sdf', 'linker_sdf']
    result_df = result_df.drop(columns=[col for col in sdf_columns if col in result_df.columns])
    
    # Reorder columns: original columns first, then fragmentation columns
    frag_columns = ['linker', 'fragments', 'distance_angstrom', 'angle_radians', 'angle_degrees', 
                    'total_energy', 'ff']
    output_columns = (original_columns or ['SMILES']) + [col for col in frag_columns if col in result_df.columns]
    result_df = result_df[output_columns]
    
    result_df.to_csv(output_csv, index=False)


def _process_smi_chunk_worker(args):
    """Worker function for processing a SMILES with its index."""
    smiles, idx, kwargs_dict = args
    try:
        seed = kwargs_dict.get('random_seed')
        if seed is not None:
            kwargs_dict['random_seed'] = seed + idx
        
        results, temp_files = process_smiles_with_geometry(smiles, **kwargs_dict)
        for result in results:
            result['SMILES'] = smiles
        return (results, temp_files, idx, None)
    except Exception as e:
        return ([], [], idx, str(e))


def process_smi_file_with_geometry(input_smi: str, output_csv: str, 
                                    num_threads: int = 1,
                                    chunk_size: int = 12800,
                                    **kwargs) -> Tuple[List[str], int]:
    """Process a .smi file: fragment molecules and compute geometry, output to CSV."""
    with open(input_smi, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    verbose = kwargs.pop('verbose', False)
    kwargs.pop('keep_temp', None)  # Remove keep_temp as it's not used by process_smiles_with_geometry
    all_temp_sdf_files = []
    total_fragments = 0
    header_written = False
    
    # Process in chunks
    total_chunks = (len(smiles_list) + chunk_size - 1) // chunk_size
    pbar = tqdm(total=len(smiles_list), desc="Processing SMILES", disable=False, unit="mol")
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(smiles_list))
        chunk_smiles = smiles_list[start_idx:end_idx]
        
        chunk_results = []
        chunk_temp_files = []
        
        if num_threads > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                tasks = [(smiles, start_idx + i, kwargs.copy()) for i, smiles in enumerate(chunk_smiles)]
                futures = {executor.submit(_process_smi_chunk_worker, task): task for task in tasks}
                
                for future in as_completed(futures):
                    results, temp_files, idx, error = future.result()
                    if error and verbose:
                        pbar.write(f"Error processing {chunk_smiles[idx - start_idx][:50]}...: {error}")
                    chunk_results.extend(results)
                    chunk_temp_files.extend(temp_files)
                    pbar.update(1)
        else:
            # Sequential processing
            for local_idx, smiles in enumerate(chunk_smiles):
                idx = start_idx + local_idx
                try:
                    seed = kwargs.get('random_seed')
                    if seed is not None:
                        kwargs['random_seed'] = seed + idx
                    
                    results, temp_files = process_smiles_with_geometry(smiles, **kwargs)
                    for result in results:
                        result['SMILES'] = smiles
                    chunk_results.extend(results)
                    chunk_temp_files.extend(temp_files)
                    pbar.update(1)
                except Exception as e:
                    if verbose:
                        pbar.write(f"Error processing {smiles[:50]}...: {e}")
                    pbar.update(1)
        
        # Write chunk results to CSV (append mode after first write)
        if chunk_results:
            chunk_df = pd.DataFrame(chunk_results)
            
            # Remove SDF file columns
            sdf_columns = ['conformer_sdf', 'fragments_sdf', 'linker_sdf']
            chunk_df = chunk_df.drop(columns=[col for col in sdf_columns if col in chunk_df.columns])
            
            # Reorder columns
            frag_columns = ['linker', 'fragments', 'distance_angstrom', 'angle_radians', 'angle_degrees',
                            'total_energy', 'ff']
            output_columns = ['SMILES'] + [col for col in frag_columns if col in chunk_df.columns]
            chunk_df = chunk_df[output_columns]
            
            # Write to CSV (append mode)
            chunk_df.to_csv(output_csv, mode='a' if header_written else 'w',
                           index=False, header=not header_written)
            header_written = True
            total_fragments += len(chunk_results)
        
        all_temp_sdf_files.extend(chunk_temp_files)
    
    pbar.close()
    
    # If no results, create empty CSV with headers
    if not header_written:
        output_columns = ['SMILES', 'linker', 'fragments', 'distance_angstrom',
                         'angle_radians', 'angle_degrees', 'total_energy', 'ff']
        pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False)
    
    return all_temp_sdf_files, total_fragments


def _process_csv_row_worker(args):
    """Worker function for processing a CSV row with its index and original row data."""
    row_dict, smiles, idx, molecule_name, kwargs_dict = args
    try:
        seed = kwargs_dict.get('random_seed')
        if seed is not None:
            kwargs_dict['random_seed'] = seed + idx
        
        # Add molecule_name to kwargs
        kwargs_dict['molecule_name'] = molecule_name
        results, temp_files = process_smiles_with_geometry(smiles, **kwargs_dict)
        
        # Add original row data to each result
        for result in results:
            for col, val in row_dict.items():
                result[col] = val
        
        return (results, temp_files, idx, None)
    except Exception as e:
        return ([], [], idx, str(e))


def process_csv_with_geometry(input_csv: str, output_csv: str, smiles_column: str = 'SMILES',
                               num_threads: int = 1,
                               molecule_name_prefix: str = None,
                               chunk_size: int = 12800,
                               **kwargs) -> Tuple[List[str], int]:
    """Process a CSV file: fragment molecules and compute geometry, output to CSV in chunks."""
    # Read first chunk to get column names
    df_chunk = pd.read_csv(input_csv, nrows=1)
    
    if smiles_column not in df_chunk.columns:
        raise ValueError(f"Column '{smiles_column}' not found in CSV. Available columns: {list(df_chunk.columns)}")
    
    # Determine molecule name prefix from input CSV filename if not provided
    if molecule_name_prefix is None:
        input_basename = os.path.splitext(os.path.basename(input_csv))[0]
        molecule_name_prefix = input_basename  # e.g., "1k_zinc" -> "1k_zinc_0", "1k_zinc_1", etc.
    
    original_columns = list(df_chunk.columns)
    verbose = kwargs.pop('verbose', False)
    kwargs.pop('keep_temp', None)  # Remove keep_temp as it's not used by process_smiles_with_geometry
    
    # Count total rows first for progress bar
    total_rows = sum(1 for _ in open(input_csv)) - 1  # Subtract header
    
    all_temp_sdf_files = []
    total_fragments = 0
    header_written = False
    
    # Process CSV in chunks of 128,000
    chunk_iter = pd.read_csv(input_csv, chunksize=chunk_size)
    pbar = tqdm(total=total_rows, desc="Processing CSV", disable=False, unit="rows")
    
    global_idx = 0
    for chunk_df in chunk_iter:
        chunk_results = []
        chunk_temp_files = []
        
        # Prepare tasks: (row_dict, smiles, idx, molecule_name, kwargs)
        tasks = []
        valid_rows = []
        for _, row in chunk_df.iterrows():
            smiles = row[smiles_column]
            if pd.isna(smiles):
                pbar.update(1)
                global_idx += 1
                continue
            
            row_dict = row.to_dict()
            molecule_name = f"{molecule_name_prefix}_{global_idx}"
            tasks.append((row_dict, smiles, global_idx, molecule_name, kwargs.copy()))
            valid_rows.append(row)
            global_idx += 1
        
        if num_threads > 1 and tasks:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(_process_csv_row_worker, task): task for task in tasks}
                
                for future in as_completed(futures):
                    results, temp_files, idx, error = future.result()
                    if error and verbose:
                        task = futures[future]
                        pbar.write(f"Error processing {task[1][:50]}...: {error}")
                    chunk_results.extend(results)
                    chunk_temp_files.extend(temp_files)
                    pbar.update(1)
        else:
            # Sequential processing
            for row_dict, smiles, idx, molecule_name, task_kwargs in tasks:
                try:
                    seed = task_kwargs.get('random_seed')
                    if seed is not None:
                        task_kwargs['random_seed'] = seed + idx
                    
                    task_kwargs['molecule_name'] = molecule_name
                    results, temp_files = process_smiles_with_geometry(smiles, **task_kwargs)
                    
                    for result in results:
                        for col, val in row_dict.items():
                            result[col] = val
                    chunk_results.extend(results)
                    chunk_temp_files.extend(temp_files)
                    pbar.update(1)
                except Exception as e:
                    if verbose:
                        pbar.write(f"Error processing {smiles[:50]}...: {e}")
                    pbar.update(1)
        
        # Write chunk results to CSV
        if chunk_results:
            chunk_df_result = pd.DataFrame(chunk_results)
            
            # Remove SDF file columns
            sdf_columns = ['conformer_sdf', 'fragments_sdf', 'linker_sdf']
            chunk_df_result = chunk_df_result.drop(columns=[col for col in sdf_columns if col in chunk_df_result.columns])
            
            # Reorder columns
            frag_columns = ['linker', 'fragments', 'distance_angstrom', 'angle_radians', 'angle_degrees',
                            'total_energy', 'ff']
            output_columns = original_columns + [col for col in frag_columns if col in chunk_df_result.columns]
            chunk_df_result = chunk_df_result[output_columns]
            
            # Write to CSV (append mode)
            chunk_df_result.to_csv(output_csv, mode='a' if header_written else 'w',
                                  index=False, header=not header_written)
            header_written = True
            total_fragments += len(chunk_results)
        
        all_temp_sdf_files.extend(chunk_temp_files)
    
    pbar.close()
    
    # If no results, create empty CSV with headers
    if not header_written:
        output_columns = original_columns + ['linker', 'fragments', 'distance_angstrom',
                                           'angle_radians', 'angle_degrees', 'total_energy', 'ff']
        pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False)
    
    return all_temp_sdf_files, total_fragments
def main():
    """CLI entry point for fragmentizing molecules."""
    parser = argparse.ArgumentParser(
        description='Fragment molecules and compute 3D geometry information. '
                    'Input can be: (1) SMILES string, (2) .smi file, or (3) CSV file.'
    )
    
    parser.add_argument('input', type=str, nargs='?',
                       help='Input: SMILES string, .smi file path, or CSV file path')
    parser.add_argument('--output-csv', type=str, default=None,
                       help='Output CSV file path (required for file inputs)')
    parser.add_argument('--smiles-column', type=str, default='SMILES',
                       help='Name of SMILES column in CSV (default: SMILES)')
    
    parser.add_argument('--linker-min', type=int, default=3,
                       help='Minimum number of heavy atoms in linker (default: 3)')
    parser.add_argument('--fragment-min', type=int, default=5,
                       help='Minimum number of heavy atoms in fragments (default: 5)')
    parser.add_argument('--min-path-length', type=int, default=2,
                       help='Minimum path length between fragments (default: 2)')
    parser.add_argument('--linker-leq-frags', action='store_true', default=True,
                       help='Linker must be <= smallest fragment (default: True)')
    parser.add_argument('--no-linker-leq-frags', dest='linker_leq_frags', action='store_false',
                       help='Disable linker <= fragment constraint')
    parser.add_argument('--num-confs', type=int, default=1,
                       help='Number of conformers to generate (default: 1)')
    parser.add_argument('--random-seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--temp-dir', type=str, default='/tmp/linkllama',
                       help='Temporary directory for SDF files (default: /tmp/linkllama)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (for SMILES mode, default: print to stdout)')
    parser.add_argument('--num-threads', type=int, default=None,
                       help='Number of threads for parallel processing (default: CPU count // 8)')
    parser.add_argument('--chunk-size', type=int, default=12800,
                       help='Number of rows to process per chunk (default: 12800)')
    parser.add_argument('--verbose', action='store_true', help='Print progress updates')
    
    args = parser.parse_args()
    
    if args.input is None:
        parser.error("Input required: SMILES string, .smi file, or CSV file")
    
    # Determine number of threads
    num_threads = args.num_threads if args.num_threads is not None else multiprocessing.cpu_count() // 8
    print(f"Using {num_threads} threads")

    # Common kwargs for processing functions
    process_kwargs = {
        'linker_min': args.linker_min,
        'fragment_min': args.fragment_min,
        'min_path_length': args.min_path_length,
        'linker_leq_frags': args.linker_leq_frags,
        'num_confs': args.num_confs,
        'random_seed': args.random_seed,
        'temp_dir': args.temp_dir,
        'verbose': args.verbose,
        'num_threads': num_threads,
        'chunk_size': args.chunk_size
    }
    
    # Check if input is a file
    if os.path.exists(args.input):
        if not args.output_csv:
            parser.error("--output-csv is required for file inputs")
        
        file_ext = os.path.splitext(args.input)[1].lower()
        
        if file_ext == '.smi':
            temp_sdf_files, num_fragments = process_smi_file_with_geometry(
                args.input, args.output_csv, **process_kwargs)
        elif file_ext == '.csv':
            molecule_name_prefix = os.path.splitext(os.path.basename(args.input))[0]
            
            temp_sdf_files, num_fragments = process_csv_with_geometry(
                args.input, args.output_csv, args.smiles_column, 
                molecule_name_prefix=molecule_name_prefix, **process_kwargs)
        else:
            parser.error(f"Unsupported file extension: {file_ext}. Use .smi or .csv")
        
        if temp_sdf_files:
            temp_dir_path = os.path.dirname(temp_sdf_files[0]) if temp_sdf_files else args.temp_dir
            print(f"\nSDF files saved: {len(temp_sdf_files)} files in {temp_dir_path}")
        print(f"\nFound {num_fragments} valid fragmentations")
    
    else:
        # Single SMILES string mode
        # Remove parameters not accepted by process_smiles_with_geometry
        single_smiles_kwargs = {k: v for k, v in process_kwargs.items() 
                               if k not in ['num_threads', 'keep_temp', 'verbose']}
        results, temp_sdf_files = process_smiles_with_geometry(args.input, **single_smiles_kwargs)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved {len(results)} fragmentations to {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        if temp_sdf_files:
            temp_dir_path = os.path.dirname(temp_sdf_files[0])
            print(f"\nSDF files saved: {len(temp_sdf_files)} files in {temp_dir_path}")
        print(f"\nFound {len(results)} valid fragmentations")


if __name__ == '__main__':
    main()

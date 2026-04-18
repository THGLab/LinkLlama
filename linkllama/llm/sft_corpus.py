#!/usr/bin/env python3
"""
Script to generate SFT training corpus from CSV file with molecular properties.
Processes CSV in chunks and generates JSONL output for fine-tuning.

Expected CSV columns:
    Required: SMILES, linker, fragments, distance_angstrom, angle_degrees
    Optional (will be calculated if missing):
        - linker_num_rotatable_bonds, linker_num_heavy_atoms, linker_topology
        - mol_num_hbd, mol_num_hba
        - mol_weight, mol_logp, mol_tpsa
        - mol_has_undesirable_pattern, mol_pains_alert, mol_reos_failed_rule
        - linker_bad_ring, linker_problematic_ring
"""

from rdkit import Chem
from rdkit import RDLogger
import pandas as pd
import json
import multiprocessing
from tqdm import tqdm
import random
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Suppress RDKit warnings (they're harmless but noisy)
RDLogger.DisableLog('rdApp.*')

# Repo root for `python linkllama/llm/sft_corpus.py` without install.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
    
from linkllama.llm.calculate_properties import process_chunk as calculate_properties_chunk
from linkllama.llm.constants import TEMPLATE

def generate_linker_type(row, **kwargs):
    """
    Generate linker type string.
    Returns: "chain", "branched", "ring-containing"
    """
    if kwargs.get('linker_type_range', True):
        return row['linker_topology'].strip()
    else:
        return ""

def generate_linker_properties(row, **kwargs):
    """
    Generate linker-specific properties string with ranges.
    Uses kwargs to probabilistically include properties (similar to property_tools.py).
    Returns: "X rotatable bonds, Y heavy atoms, [chain/branched/ring-containing]"
    
    Args:
        row: dict with pre-calculated properties (required)
        **kwargs: Property inclusion flags
    """
    specifications = []
    
    # Read pre-calculated values from row (already validated in calculate_properties.py)
    num_rotb = float(row['linker_num_rotatable_bonds'])
    num_heavy = int(row['linker_num_heavy_atoms'])
    
    # Include rotatable bonds if specified (default: True, 50% chance)
    if kwargs.get('rotb_range', True):
        rotb_ranges = []
        if num_rotb >= 4: rotb_ranges.append('>= 4')
        if num_rotb >= 3: rotb_ranges.append('>= 3')
        if num_rotb >= 2: rotb_ranges.append('>= 2')
        if num_rotb >= 1: rotb_ranges.append('>= 1')
        if num_rotb == 0: rotb_ranges.append('0')
        specifications.append(random.choice(rotb_ranges) + ' rotatable bonds')
    
    # Include heavy atoms if specified (default: True, 50% chance)
    if kwargs.get('heavy_atoms_range', True):
        heavy_ranges = []
        if num_heavy >= 10: heavy_ranges.append('>= 10')
        if num_heavy >= 7: heavy_ranges.append('>= 7')
        if num_heavy >= 6: heavy_ranges.append('>= 6')
        if num_heavy >= 5: heavy_ranges.append('>= 5')
        if num_heavy >= 4: heavy_ranges.append('>= 4')
        if num_heavy >= 3: heavy_ranges.append('>= 3')
    
        specifications.append(random.choice(heavy_ranges) + ' heavy atoms')
    
    # Always include topology
    if len(specifications) > 0:
        return "with " + ", ".join(specifications)
    else:
        return ""

def generate_molecule_properties(row, **kwargs):
    """
    Generate properties for the full molecule (fragments + linker) with ranges.
    Uses kwargs to probabilistically include properties (similar to property_tools.py).
    
    Args:
        row: dict with pre-calculated properties (required)
        **kwargs: Property inclusion flags
    """
    specifications = []
    
    # Read pre-calculated values from row (already validated in calculate_properties.py)
    num_hbd = int(row['mol_num_hbd'])
    num_hba = int(row['mol_num_hba'])
    mol_weight = float(row['mol_weight'])
    logp = float(row['mol_logp'])
    tpsa = float(row['mol_tpsa'])
    
    # Include HBD if specified (default: True, 50% chance)
    if kwargs.get('hbd_range', True):
        hbd_ranges = []
        if num_hbd > 5: hbd_ranges.append('> 5')
        if num_hbd <= 5: hbd_ranges.append('<= 5')
        if num_hbd <= 3: hbd_ranges.append('<= 3')
        if num_hbd <= 2: hbd_ranges.append('<= 2')
        if num_hbd <= 1: hbd_ranges.append('<= 1')
        specifications.append(random.choice(hbd_ranges) + ' H-bond donors')
    
    # Include HBA if specified (default: True, 50% chance)
    if kwargs.get('hba_range', True):
        hba_ranges = []
        if num_hba > 10: hba_ranges.append('> 10')
        if num_hba <= 10: hba_ranges.append('<= 10')
        if num_hba <= 7: hba_ranges.append('<= 7')
        if num_hba <= 5: hba_ranges.append('<= 5')
        if num_hba <= 3: hba_ranges.append('<= 3')
        specifications.append(random.choice(hba_ranges) + ' H-bond acceptors')
    
    # Include MW if specified (default: True, 50% chance)
    if kwargs.get('mw_range', True):
        mw_ranges = []
        if mol_weight > 700: mw_ranges.append('> 700')    
        if mol_weight <= 700: mw_ranges.append('<= 700')
        if mol_weight <= 600: mw_ranges.append('<= 600')
        if mol_weight <= 500: mw_ranges.append('<= 500')
        if mol_weight <= 400: mw_ranges.append('<= 400')
        if mol_weight <= 300: mw_ranges.append('<= 300')
        specifications.append(random.choice(mw_ranges) + ' Molecular weight')
    
    # Include LogP if specified (default: True, 50% chance)
    if kwargs.get('logp_range', True):
        logp_ranges = []
        if logp < 0: logp_ranges.append('< 0')
        else:
            if logp > 6: logp_ranges.append('> 6')
            if logp <= 6: logp_ranges.append('<= 6')
            if logp <= 5: logp_ranges.append('<= 5')
            if logp <= 4: logp_ranges.append('<= 4')
            if logp <= 3: logp_ranges.append('<= 3')
        specifications.append(random.choice(logp_ranges) + ' LogP')
    
    # Include TPSA if specified (default: True, 50% chance)
    if kwargs.get('tpsa_range', True):
        tpsa_ranges = []
        if tpsa > 200: tpsa_ranges.append('> 200')
        if tpsa <= 200: tpsa_ranges.append('<= 200')
        if tpsa <= 140: tpsa_ranges.append('<= 140')
        if tpsa <= 90: tpsa_ranges.append('<= 90')
        specifications.append(random.choice(tpsa_ranges) + ' TPSA')
    
    if len(specifications) > 0:
        return " And it should have the following properties: " + ", ".join(specifications) + "."
    else:
        return ""


def generate_fragment_info(fragments_smi, distance, angle):
    """
    Format fragment information with geometry.
    """
    frags = fragments_smi.split('.')
    frag1 = frags[0].strip()
    frag2 = frags[1].strip()
    
    return f"Fragment 1 (SMILES: {frag1}) and Fragment 2 (SMILES: {frag2}). The distance between the attachment points is {distance:.2f} Angstroms, and the angle between them is {angle:.2f} degrees."


def reasonability_pass_masks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized version of the same pass/fail rules as get_reasonability_and_reasoning.

    Clean (pass) semantics match that function:
    - REOS: no failed Dundee rule → stored value is '' or str is 'nan' after strip (CSV NaN).
    - Problematic ring: check_ring_system returned '' or missing → same '' / 'nan' strip rule.
    - Undesirable / PAINS / bad ring: explicit False on the boolean columns (NaN does not pass).

    Returns a DataFrame with boolean columns pass_undesirable, pass_pains, pass_reos,
    pass_bad_ring, pass_problematic_ring, pass_all.
    """
    mol_reos_rule = df["mol_reos_failed_rule"].astype(str).str.strip()
    linker_problematic = df["linker_problematic_ring"].astype(str).str.strip()
    pass_reos = mol_reos_rule.isin(("", "nan"))
    pass_problematic_ring = linker_problematic.isin(("", "nan"))
    pass_undesirable = df["mol_has_undesirable_pattern"].eq(False)
    pass_pains = df["mol_pains_alert"].eq(False)
    pass_bad_ring = df["linker_bad_ring"].eq(False)
    pass_all = (
        pass_undesirable
        & pass_pains
        & pass_reos
        & pass_bad_ring
        & pass_problematic_ring
    )
    return pd.DataFrame(
        {
            "pass_undesirable": pass_undesirable,
            "pass_pains": pass_pains,
            "pass_reos": pass_reos,
            "pass_bad_ring": pass_bad_ring,
            "pass_problematic_ring": pass_problematic_ring,
            "pass_all": pass_all,
        }
    )


def get_reasonability_and_reasoning(row):
    """
    From row checks, compute overall reasonability and reasoning text in one pass.
    Assumes all properties are already calculated and validated in calculate_properties.py.

    Returns:
        tuple: (overall_reasonability, reasoning_text)
        - overall_reasonability: "reasonable" if all checks pass, "unreasonable" otherwise
        - reasoning_text: human-readable summary of each check (linker bad/problematic rings,
          undesirable SMARTS, PAINS, REOS)
    """
    linker_bad_ring = bool(row['linker_bad_ring'])
    linker_problematic = str(row['linker_problematic_ring']).strip()
    mol_has_undesirable = bool(row['mol_has_undesirable_pattern'])
    mol_has_pains = bool(row['mol_pains_alert'])
    mol_reos_rule = str(row['mol_reos_failed_rule']).strip()

    reasoning_parts = []
    reasoning_parts.append(f"Linker bad rings: {'fail' if linker_bad_ring else 'pass'}")
    if linker_problematic in ('', 'nan'):
        reasoning_parts.append("Linker problematic ring: absent")
    else:
        reasoning_parts.append(f"Linker problematic ring: {linker_problematic}")
    reasoning_parts.append(f"Undesirable SMARTS: {'fail' if mol_has_undesirable else 'pass'}")
    reasoning_parts.append(f"PAINS: {'fail' if mol_has_pains else 'pass'}")
    if mol_reos_rule in ('', 'nan'):
        reasoning_parts.append("REOS failed rule: absent")
    else:
        reasoning_parts.append(f"REOS failed rule: {mol_reos_rule}")

    if linker_bad_ring or linker_problematic not in ('', 'nan') or mol_has_undesirable or mol_has_pains or mol_reos_rule not in ('', 'nan'):
        overall_reasonability = "unreasonable"
    else:
        overall_reasonability = "reasonable"

    reasoning_text = ". ".join(reasoning_parts) + "."
    return overall_reasonability, reasoning_text


def create_linker_training_pair(row):
    """
    Main function to create a training pair from a CSV row.
    Assumes all properties are already calculated and present in the row.
    
    Returns None if row has errors or exceptions, which will cause it to be skipped.
    """
    try:
        # Parse row data (already validated in calculate_properties.py)
        full_smiles = str(row['SMILES'])
        linker_smi = str(row['linker'])
        fragments_smi = str(row['fragments'])
        distance = float(row['distance_angstrom'])
        angle = float(row['angle_degrees'])
        
        # Generate components
        fragment_info = generate_fragment_info(fragments_smi, distance, angle)
        
        # Randomly decide which properties to include (50% chance each, similar to make_sft_data.ipynb)
        linker_kwargs = {
            'rotb_range': (0.5 > random.random()),
            'heavy_atoms_range': (0.5 > random.random()),
            'linker_type_range': (0.5 > random.random())
        }
        molecule_kwargs = {
            'hbd_range': (0.5 > random.random()),
            'hba_range': (0.5 > random.random()),
            'mw_range': (0.5 > random.random()),
            'logp_range': (0.5 > random.random()),
            'tpsa_range': (0.5 > random.random())
        }
        
        # Use pre-calculated properties from row (already validated in calculate_properties.py)
        linker_properties = generate_linker_properties(row, **linker_kwargs)
        linker_type = generate_linker_type(row, **linker_kwargs)
        molecule_properties = generate_molecule_properties(row, **molecule_kwargs)
        
        overall_reasonability, reasoning = get_reasonability_and_reasoning(row)

        # Fill in template
        template = TEMPLATE.copy()
        input_text = template['input']
        input_text = input_text.replace('FRAGMENT_INFO', fragment_info)
        
        # Handle linker properties
        input_text = input_text.replace('LINKER_TYPE', linker_type)
        input_text = input_text.replace('LINKER_PROPERTIES', linker_properties)
        input_text = input_text.replace('MOLECULE_PROPERTIES', molecule_properties)
        input_text = input_text.replace('REASONABILITY', overall_reasonability)
        # Clean up any double spaces
        input_text = ' '.join(input_text.split())
        
        # Create output JSON (include check_reasonability for every row)
        output_dict = {
            "linker": linker_smi,
            "reasoning": reasoning
        }
        output_json = json.dumps(output_dict)
        
        return {
            "instruction": template['instruction'],
            "input": input_text,
            "output": output_json
        }
    
    except Exception as e:
        print(f"Error processing row: {e}")
        return None


def parallelize_processing(df: pd.DataFrame, num_processes: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Process dataframe in parallel.
    """
    if num_processes is None:
        num_cores = multiprocessing.cpu_count()
    else:
        num_cores = num_processes
    print(f"Number of cores: {num_cores}")
    
    # Convert dataframe rows to list of dicts (to avoid Series boolean ambiguity issues)
    rows = [row.to_dict() for _, row in df.iterrows()]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(create_linker_training_pair, rows), total=len(rows)))
    
    # Remove None values
    results = [r for r in results if r is not None]
    return results


def main(
    input_csv_path: Path,
    output_jsonl_path: Optional[Path] = None,
    num_processes: Optional[int] = None,
    chunk_size: int = 12800
) -> None:
    """
    Main function to process CSV and generate JSONL output.
    
    Checks if CSV has all required property columns at the start.
    If not, calculates properties for each chunk before processing.
    If yes, skips calculation and goes directly to prompt preparation.
    
    Args:
        input_csv_path: Path to input CSV file
        output_jsonl_path: Path to output JSONL file (default: input_csv_path with .jsonl extension)
        num_processes: Number of parallel processes (default: CPU count // 4)
        chunk_size: Process CSV in chunks of this size (default: 12800)
    """
    # Set defaults
    if output_jsonl_path is None:
        output_jsonl_path = input_csv_path.with_suffix('.jsonl')
    if num_processes is None:
        num_processes = multiprocessing.cpu_count() // 4
    
    # Check if CSV has all required property columns
    print(f"Checking CSV columns: {input_csv_path}")
    header_df = pd.read_csv(str(input_csv_path), nrows=0)
    existing_columns = set(header_df.columns)
    
    # Define all required property columns
    required_property_cols = {
        'linker_num_rotatable_bonds', 'linker_num_heavy_atoms', 'linker_topology',
        'mol_num_hbd', 'mol_num_hba', 'mol_weight', 'mol_logp', 'mol_tpsa',
        'mol_has_undesirable_pattern', 'mol_pains_alert', 'mol_reos_failed_rule',
        'linker_bad_ring', 'linker_problematic_ring'
    }
    
    # Check if all required columns exist
    missing_cols = required_property_cols - existing_columns
    needs_calculation = len(missing_cols) > 0
    
    if needs_calculation:
        print(f"Missing property columns: {missing_cols}")
        print(f"Will calculate properties for each chunk before processing...")
    else:
        print(f"All required property columns found. Skipping calculation.")
    
    print(f"Processing CSV in chunks of {chunk_size} rows with {num_processes} processes...")
    print(f"Input: {input_csv_path}")
    print(f"Output: {output_jsonl_path}")
    
    total_processed = 0
    total_generated = 0
    
    with open(output_jsonl_path, 'w') as f:
        for chunk_df in pd.read_csv(str(input_csv_path), chunksize=chunk_size):
            print(f"Processing chunk: {len(chunk_df)} rows (total so far: {total_processed})")
            
            # Case 1: Calculate properties if needed
            original_chunk_size = len(chunk_df)
            if needs_calculation:
                print(f"  Calculating properties for chunk...")
                chunk_df = calculate_properties_chunk(chunk_df, num_workers=num_processes, verbose=False)
                if len(chunk_df) == 0:
                    print(f"  All rows in chunk were dropped due to errors, skipping...")
                    total_processed += original_chunk_size
                    continue
            
            # Case 2: Properties are already present, go directly to prompt preparation
            print(f"  Generating training pairs...")
            training_pairs = parallelize_processing(chunk_df, num_processes=num_processes)
            
            # Write immediately to avoid memory buildup
            for pair in training_pairs:
                f.write(json.dumps(pair) + '\n')
            
            total_processed += original_chunk_size  # Track original chunk size, not filtered size
            total_generated += len(training_pairs)
    
    print(f"Processed {total_processed} rows, generated {total_generated} training pairs")
    print("Done!")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate SFT training corpus from CSV file with molecular properties. "
            "Processes CSV in chunks and generates JSONL output for fine-tuning. "
            "Expects CSV columns: SMILES, linker, fragments, distance_angstrom, angle_degrees, "
            "and optionally pre-calculated property columns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s input.csv\n"
            "  %(prog)s input.csv -o output.jsonl\n"
            "  %(prog)s input.csv --num-processes 8 --chunk-size 10000\n"
            "  %(prog)s input.csv -o output.jsonl -p 4 -c 5000\n"
        )
    )
    
    parser.add_argument(
        'input_csv',
        type=Path,
        help='Path to input CSV file with molecular data'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        dest='output_jsonl',
        metavar='OUTPUT',
        help='Path to output JSONL file (default: input_csv with .jsonl extension)'
    )
    
    parser.add_argument(
        '-p', '--num-processes',
        type=int,
        default=None,
        dest='num_processes',
        metavar='N',
        help=(
            'Number of parallel processes. '
            f'Default: cpu_count() // 4 ({multiprocessing.cpu_count() // 4})'
        )
    )
    
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=12800,
        metavar='N',
        help='Number of rows to process per chunk (default: 12800)'
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
    
    # Validate num_processes
    if args.num_processes is not None and args.num_processes <= 0:
        parser.error(f"num_processes must be positive, got: {args.num_processes}")
    
    max_processes = multiprocessing.cpu_count()
    if args.num_processes is not None and args.num_processes > max_processes:
        parser.error(
            f"num_processes ({args.num_processes}) exceeds available CPUs ({max_processes})"
        )
    
    # Check if output directory exists, create if not
    if args.output_jsonl is not None:
        output_dir = args.output_jsonl.parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        input_csv_path=args.input_csv,
        output_jsonl_path=args.output_jsonl,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size
    )


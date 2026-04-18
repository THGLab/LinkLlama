import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os

def read_sdf_efficient(sdf_path, output_csv=None):
    """
    Efficiently read SDF file and extract metadata with progress bar.
    
    Args:
        sdf_path: Path to the SDF file
        output_csv: Path for output CSV (optional, defaults to same name as SDF)
    
    Returns:
        DataFrame with SMILES and metadata
    """
    
    # Determine output path
    if output_csv is None:
        output_csv = sdf_path.replace('.sdf', '.csv')
    
    # First pass: count molecules for progress bar
    print(f"Counting molecules in {os.path.basename(sdf_path)}...")
    suppl = Chem.SDMolSupplier(sdf_path)
    total_mols = len(suppl)
    
    # Second pass: extract data
    print(f"Processing {total_mols} molecules...")
    
    data = []
    
    for mol in tqdm(suppl, total=total_mols, desc="Reading SDF"):
        if mol is None:
            continue
        
        # Get SMILES
        smiles = Chem.MolToSmiles(mol)
        
        # Extract all properties
        props = mol.GetPropsAsDict()
        
        # Build row with SMILES first, then all properties
        row = {'SMILES': smiles}
        row.update(props)
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} molecules to {output_csv}")
    
    return df

if __name__ == "__main__":
    # Process the large SDF file
    sdf_file = "/pscratch/sd/k/kysun/LLM-research/leadoptllama/data/enamine_24k_comprehensive_linker/Enamine_Comprehensive_Linkers_23978cmpds_20250913.sdf"
    df = read_sdf_efficient(sdf_file)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())


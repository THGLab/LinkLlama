"""
3D conformer generation from SMILES strings.

This module generates 3D conformers for molecules using RDKit's
embedding and optimization methods.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, List, Tuple
import random


def generate_conformer(mol: Chem.Mol, 
                       num_confs: int = 1,
                       random_seed: Optional[int] = None,
                       use_etkdg: bool = True,
                       optimize: bool = True) -> Tuple[Optional[Chem.Mol], Optional[float], Optional[str]]:
    """
    Generate 3D conformer(s) for a molecule.
    
    Args:
        mol: RDKit molecule object
        num_confs: Number of conformers to generate
        random_seed: Random seed for reproducibility
        use_etkdg: Use ETKDG (Experimental Torsion-angle preference with Distance Geometry)
        optimize: Optimize conformers with MMFF
        
    Returns:
        Tuple of (molecule with conformer(s), energy in kcal/mol, force_field_name), 
        or (None, None, None) if generation failed.
        force_field_name is either 'MMFF' or 'UFF' depending on which succeeded.
    """
    if mol is None:
        return None, None, None
    
    try:
        # Add hydrogens
        mol_with_hs = Chem.AddHs(mol)
        
        # Generate conformers
        # Build kwargs dict, only include randomSeed if not None
        embed_kwargs = {
            'numConfs': num_confs,
            'useExpTorsionAnglePrefs': True,
            'useBasicKnowledge': True,
        }
        if random_seed is not None:
            embed_kwargs['randomSeed'] = random_seed
        
        if use_etkdg:
            # Try with useSmallRingTorsions first
            try:
                embed_kwargs['useSmallRingTorsions'] = True
                cids = AllChem.EmbedMultipleConfs(mol_with_hs, **embed_kwargs)
            except (TypeError, AttributeError):
                # Fallback without useSmallRingTorsions
                embed_kwargs.pop('useSmallRingTorsions', None)
                try:
                    cids = AllChem.EmbedMultipleConfs(mol_with_hs, **embed_kwargs)
                except (TypeError, AttributeError):
                    # Final fallback - just numConfs
                    cids = AllChem.EmbedMultipleConfs(mol_with_hs, numConfs=num_confs)
        else:
            # Basic embedding
            cids = AllChem.EmbedMultipleConfs(mol_with_hs, **embed_kwargs)
        
        if len(cids) == 0:
            return None, None, None
        
        # Optimize conformers and calculate energy
        energy = None
        force_field_name = None
        
        if optimize:
            for cid in cids:
                try:
                    result = AllChem.MMFFOptimizeMolecule(mol_with_hs, confId=cid)
                    # result is 0 if successful, -1 if failed
                    if result == 0:
                        # MMFF succeeded, calculate energy
                        try:
                            mp = AllChem.MMFFGetMoleculeProperties(mol_with_hs)
                            if mp is not None:
                                ff = AllChem.MMFFGetMoleculeForceField(mol_with_hs, mp, confId=cid)
                                if ff is not None:
                                    energy = ff.CalcEnergy()
                                    force_field_name = 'MMFF'
                        except:
                            pass
                    else:
                        # If MMFF fails, try UFF
                        try:
                            result_uff = AllChem.UFFOptimizeMolecule(mol_with_hs, confId=cid)
                            if result_uff == 0:
                                # UFF succeeded, calculate energy
                                try:
                                    ff = AllChem.UFFGetMoleculeForceField(mol_with_hs, confId=cid)
                                    if ff is not None:
                                        energy = ff.CalcEnergy()
                                        force_field_name = 'UFF'
                                except:
                                    pass
                        except:
                            pass
                except:
                    # If optimization fails completely, just continue with unoptimized conformer
                    pass
        
        return mol_with_hs, energy, force_field_name
    
    except Exception as e:
        # Uncomment for debugging:
        # print(f"Error in generate_conformer: {e}")
        # traceback.print_exc()
        return None, None, None


def generate_conformer_from_smiles(smiles: str,
                                    num_confs: int = 1,
                                    random_seed: Optional[int] = None,
                                    use_etkdg: bool = True,
                                    optimize: bool = True) -> Tuple[Optional[Chem.Mol], Optional[float], Optional[str]]:
    """
    Generate 3D conformer(s) from a SMILES string.
    
    Args:
        smiles: SMILES string
        num_confs: Number of conformers to generate
        random_seed: Random seed for reproducibility
        use_etkdg: Use ETKDG method
        optimize: Optimize conformers with MMFF
        
    Returns:
        Tuple of (molecule with conformer(s), energy in kcal/mol, force_field_name), 
        or (None, None, None) if generation failed.
        force_field_name is either 'MMFF' or 'UFF' depending on which succeeded.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    
    return generate_conformer(mol, num_confs=num_confs, random_seed=random_seed,
                             use_etkdg=use_etkdg, optimize=optimize)


def generate_randomized_conformer(smiles: str,
                                   num_confs: int = 10,
                                   random_seed: Optional[int] = None) -> Tuple[Optional[Chem.Mol], Optional[float], Optional[str]]:
    """
    Generate a randomized 3D conformer from SMILES.
    
    This function generates multiple conformers and returns one randomly,
    which helps with geometric diversity in the dataset.
    
    Args:
        smiles: SMILES string
        num_confs: Number of conformers to generate (one will be selected randomly)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (molecule with a randomly selected conformer, energy in kcal/mol, force_field_name), 
        or (None, None, None) if generation failed.
        force_field_name is either 'MMFF' or 'UFF' depending on which succeeded.
    """
    mol, _, _ = generate_conformer_from_smiles(smiles, num_confs=num_confs, 
                                        random_seed=random_seed)
    
    if mol is None or mol.GetNumConformers() == 0:
        return None, None, None
    
    # Select a random conformer
    if random_seed is not None:
        random.seed(random_seed)
    conf_id = random.choice(list(range(mol.GetNumConformers())))
    
    # Create new molecule with only the selected conformer
    mol_with_conf = Chem.Mol(mol)
    mol_with_conf.RemoveAllConformers()
    conf = mol.GetConformer(conf_id)
    mol_with_conf.AddConformer(conf)
    
    # Calculate energy for the selected conformer
    energy = None
    force_field_name = None
    try:
        # Try MMFF first
        mp = AllChem.MMFFGetMoleculeProperties(mol_with_conf)
        if mp is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol_with_conf, mp, confId=0)
            if ff is not None:
                energy = ff.CalcEnergy()
                force_field_name = 'MMFF'
    except:
        pass
    
    if energy is None:
        # If MMFF failed, try UFF
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol_with_conf, confId=0)
            if ff is not None:
                energy = ff.CalcEnergy()
                force_field_name = 'UFF'
        except:
            pass
    
    return mol_with_conf, energy, force_field_name


def save_conformers_to_sdf(mols: List[Chem.Mol], output_path: str):
    """
    Save molecules with conformers to an SDF file.
    
    Args:
        mols: List of RDKit molecules with conformers
        output_path: Path to output SDF file
    """
    writer = Chem.SDWriter(output_path)
    for mol in mols:
        if mol is not None and mol.GetNumConformers() > 0:
            writer.write(mol)
    writer.close()


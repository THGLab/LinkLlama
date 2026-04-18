"""
3D geometry calculations for linker design.

This module computes geometric properties (distance and angle) between
exit vectors of fragments using 3D conformer information.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product
from typing import Tuple, Optional, List


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return the unit vector of the vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def find_fragment_linker_matches(mol: Chem.Mol, 
                                  smi_linker: str, 
                                  smi_frags: str) -> Tuple[Optional[tuple], Optional[tuple]]:
    """
    Find matching atom indices for fragments and linker in a molecule.
    
    Args:
        mol: Molecule with 3D conformer
        smi_linker: SMILES string of linker (with dummy atoms)
        smi_frags: SMILES string of fragments (with dummy atoms)
        
    Returns:
        Tuple of (frag_match, linker_match) atom index tuples, or (None, None) if not found
    """
    frags = Chem.MolFromSmiles(smi_frags)
    linker = Chem.MolFromSmiles(smi_linker)
    
    if frags is None or linker is None:
        return None, None
    
    # Include dummy in query
    qp = Chem.AdjustQueryParameters()
    qp.makeDummiesQueries = True
    
    # Align to frags and linker
    qfrag = Chem.AdjustQueryProperties(frags, qp)
    frags_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
    qlinker = Chem.AdjustQueryProperties(linker, qp)
    linker_matches = list(mol.GetSubstructMatches(qlinker, uniquify=False))
    
    if not frags_matches or not linker_matches:
        return None, None
    
    # Find matching combination
    for f_match, l_match in product(frags_matches, linker_matches):
        # Check if match covers all atoms
        f_match_clean = [idx for num, idx in enumerate(f_match) 
                        if frags.GetAtomWithIdx(num).GetAtomicNum() != 0]
        l_match_clean = [idx for num, idx in enumerate(l_match) 
                        if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 
                        and idx not in f_match_clean]
        
        if len(set(f_match_clean + l_match_clean)) == mol.GetNumHeavyAtoms():
            return f_match, l_match
    
    return None, None


def extract_substructure_with_3d(mol: Chem.Mol, 
                                  query_smiles: str, 
                                  atom_indices: List[int]) -> Optional[Chem.Mol]:
    """
    Extract a substructure from a molecule with 3D coordinates preserved.
    Preserves dummy atoms from the query molecule.
    
    Args:
        mol: Full molecule with 3D conformer
        query_smiles: SMILES string of the substructure (with dummy atoms)
        atom_indices: List of atom indices in the full molecule that match the substructure
        
    Returns:
        Molecule with the substructure and 3D coordinates, or None if extraction fails
    """
    try:
        # Create query molecule (this has dummy atoms)
        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            return None
        
        # Use the provided match indices
        # Since we're in fragmentation context, dummy atoms are always present
        # The match should already be provided correctly, but verify
        match = atom_indices
        if len(match) != query_mol.GetNumAtoms():
            # Try to get match if indices don't match
            # Always use dummy-aware query since dummy atoms are always present
            qp = Chem.AdjustQueryParameters()
            qp.makeDummiesQueries = True
            qquery = Chem.AdjustQueryProperties(query_mol, qp)
            matches = list(mol.GetSubstructMatches(qquery, uniquify=False))
            if matches:
                match = matches[0]
            else:
                return None
        
        # Create editable molecule
        new_mol = Chem.RWMol()
        
        # Map old atom indices to new atom indices
        atom_map = {}
        conf = mol.GetConformer()
        
        # Add atoms with coordinates (including dummy atoms from query)
        for query_idx, old_idx in enumerate(match):
            if query_idx >= query_mol.GetNumAtoms():
                continue
                
            query_atom = query_mol.GetAtomWithIdx(query_idx)
            old_atom = mol.GetAtomWithIdx(old_idx)
            
            # Use atomic number from query (to preserve dummy atoms)
            atomic_num = query_atom.GetAtomicNum()
            new_atom = Chem.Atom(atomic_num)
            
            # Copy properties if not dummy
            if atomic_num != 0:
                new_atom.SetFormalCharge(old_atom.GetFormalCharge())
                # Don't copy aromatic flags - dummy atoms cause kekulization issues with aromatic systems
                # We'll clear all aromatic flags at the end anyway
            
            new_mol.AddAtom(new_atom)
            new_idx = new_mol.GetNumAtoms() - 1
            atom_map[query_idx] = new_idx
            
            # Set 3D coordinates (use position from full molecule)
            pos = conf.GetAtomPosition(old_idx)
            if new_mol.GetNumConformers() == 0:
                new_conf = Chem.Conformer(new_mol.GetNumAtoms())
                new_mol.AddConformer(new_conf)
            new_mol.GetConformer(0).SetAtomPosition(new_idx, pos)
        
        # Add bonds from query molecule
        # In fragmentation context: dummy atoms are always present
        # Rules: 1) Dummies don't connect to each other, 2) Dummy-real bonds are always SINGLE
        for bond in query_mol.GetBonds():
            begin_query_idx = bond.GetBeginAtomIdx()
            end_query_idx = bond.GetEndAtomIdx()
            
            if begin_query_idx in atom_map and end_query_idx in atom_map:
                begin_atom = query_mol.GetAtomWithIdx(begin_query_idx)
                end_atom = query_mol.GetAtomWithIdx(end_query_idx)
                begin_is_dummy = begin_atom.GetAtomicNum() == 0
                end_is_dummy = end_atom.GetAtomicNum() == 0
                
                # Skip bonds between two dummy atoms (dummies don't connect to each other)
                if begin_is_dummy and end_is_dummy:
                    continue
                
                # Determine bond type
                if begin_is_dummy or end_is_dummy:
                    # Bond between dummy and real atom: always SINGLE
                    bond_type = Chem.BondType.SINGLE
                else:
                    # Bond between two real atoms
                    # Convert aromatic bonds to single bonds to avoid kekulization issues
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        bond_type = Chem.BondType.SINGLE
                    else:
                        bond_type = bond.GetBondType()
                
                new_mol.AddBond(atom_map[begin_query_idx], 
                              atom_map[end_query_idx], 
                              bond_type)
        
        result = new_mol.GetMol()
        
        # Always clear aromatic flags when dummy atoms are present (which is always in fragmentation)
        # This prevents kekulization errors when saving to SDF
        for atom in result.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in result.GetBonds():
            bond.SetIsAromatic(False)
        
        # Try to sanitize, but it may fail with dummy atoms - that's OK
        if result.GetNumAtoms() > 0:
            try:
                Chem.SanitizeMol(result)
            except:
                pass  # Expected to fail with dummy atoms sometimes, that's OK
        
        return result
    
    except Exception:
        return None


def compute_distance_and_angle(mol: Chem.Mol, 
                                smi_linker: str, 
                                smi_frags: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute distance and angle between exit vectors of fragments.
    
    Args:
        mol: Molecule with 3D conformer
        smi_linker: SMILES string of linker
        smi_frags: SMILES string of fragments (with dummy atoms)
        
    Returns:
        Tuple of (distance, angle) in Angstroms and radians, or (None, None) if failed
    """
    try:
        frags = Chem.MolFromSmiles(smi_frags)
        linker = Chem.MolFromSmiles(smi_linker)
        
        if frags is None or linker is None:
            return None, None
        
        # Include dummy in query
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries = True
        
        # Renumber based on frags (incl. dummy atoms)
        aligned_mols = []
        
        # Find matching atom indices
        frag_match, linker_match = find_fragment_linker_matches(mol, smi_linker, smi_frags)
        
        if frag_match is None or linker_match is None:
            return None, None
        
        # Build renumbering index
        sub_idx = list(frag_match)
        # Add linker indices to end
        sub_idx += [idx for num, idx in enumerate(linker_match) 
                   if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 
                   and idx not in sub_idx]
        
        nodes_to_keep = list(range(len(frag_match)))
        
        aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))
        aligned_mols.append(frags)
        
        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        
        for i, mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms() + 2))
            for idx in dummy_idx:
                if idx in sub_idx:
                    sub_idx.remove(idx)
                    sub_idx.append(idx)
            
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms() + 2))
            
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)
        
        # Get exit vectors
        exit_vectors = []
        linker_atom_idx = []
        
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())
                linker_atom_idx.append(atom.GetIdx())
        
        if len(exit_vectors) != 2 or len(linker_atom_idx) != 2:
            return None, None
        
        # Get coordinates
        conf = aligned_mols[0].GetConformer()
        exit_coords = [np.array(conf.GetAtomPosition(exit)) for exit in exit_vectors]
        linker_coords = [np.array(conf.GetAtomPosition(linker_atom)) 
                        for linker_atom in linker_atom_idx]
        
        # Get angle
        v1_u = unit_vector(linker_coords[0] - exit_coords[0])
        v2_u = unit_vector(linker_coords[1] - exit_coords[1])
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        
        # Get distance
        distance = np.linalg.norm(exit_coords[0] - exit_coords[1])
        
        return distance, angle
    
    except Exception as e:
        return None, None


def compute_distance_and_angle_dataset(fragmentations: List[List[str]], 
                                        path_to_conformers: str,
                                        dataset: str = None,
                                        verbose: bool = False) -> Tuple[List, List, List, Tuple]:
    """
    Compute distance and angle for a dataset of fragmentations.
    
    Args:
        fragmentations: List of fragmentations, each as [cid, core, chains]
        path_to_conformers: Path to SDF file with conformers
        dataset: Dataset name ("ZINC" or "CASF") for name extraction
        verbose: Print progress updates
        
    Returns:
        Tuple of (fragmentations_new, distances, angles, (fail_count, fail_count_conf, fails))
    """
    # Load conformers
    conformers = Chem.SDMolSupplier(path_to_conformers)
    
    # Convert dataset to dictionary
    dataset_dict = {}
    for toks in fragmentations:
        if len(toks) < 3:
            continue
        key = toks[0]  # cid
        if key in dataset_dict:
            dataset_dict[key].append([toks[1], toks[2] + '.' + (toks[3] if len(toks) > 3 else '')])
        else:
            dataset_dict[key] = [[toks[1], toks[2] + '.' + (toks[3] if len(toks) > 3 else '')]]
    
    # Initialize placeholders for results
    fragmentations_new = []
    distances = []
    angles = []
    
    # Record number of failures
    fail_count = 0
    fail_count_conf = 0
    fails = []
    
    # Loop over conformers
    for count, mol in enumerate(conformers):
        if mol is not None:
            # Get molecule name
            if dataset == "ZINC":
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else Chem.MolToSmiles(mol)
            elif dataset == "CASF":
                mol_name = Chem.MolToSmiles(mol)
            else:
                mol_name = Chem.MolToSmiles(mol)
            
            if mol_name in dataset_dict:
                # Loop over all fragmentations of this mol
                for fragments in dataset_dict[mol_name]:
                    dist, ang = compute_distance_and_angle(mol, fragments[0], fragments[1])
                    if dist is not None and ang is not None:
                        fragmentations_new.append([mol_name] + fragments)
                        distances.append(dist)
                        angles.append(ang)
                    else:
                        fails.append([mol_name] + fragments)
                        fail_count += 1
        else:
            fail_count_conf += 1
        
        if verbose and count % 1000 == 0:
            print(f"\rMol: {count}", end='')
    
    if verbose:
        print("\rDone")
        print(f"Fail count conf: {fail_count_conf}")
    
    return fragmentations_new, distances, angles, (fail_count, fail_count_conf, fails)


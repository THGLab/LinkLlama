"""
Molecule fragmentation utilities for linker design.

This module provides functions to fragment molecules into cores and linkers
using RDKit's Matched Molecular Pair Analysis (MMPA).

# adapted from https://github.com/oxpig/DeLinker/data/frag_utils.py
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdMMPA
from typing import List, Set, Tuple, Optional
import numpy as np


def remove_dummys(smi_string: str) -> str:
    """Remove dummy atoms (*) from SMILES string."""
    mol = Chem.MolFromSmiles(smi_string)
    if mol is None:
        return smi_string
    mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), True)[0]
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def remove_dummys_mol(smi_string: str) -> Chem.Mol:
    """Remove dummy atoms (*) from SMILES and return molecule object."""
    mol = Chem.MolFromSmiles(smi_string)
    if mol is None:
        return None
    mol = AllChem.ReplaceSubstructs(mol, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), True)[0]
    return Chem.RemoveHs(mol)


def fragment_mol(smi: str, cid: str, pattern: str = "[#6+0;!$(*=,#[!#6])]!@!=!#[*]") -> Set[str]:
    """
    Fragment a molecule using MMPA (Matched Molecular Pair Analysis).
    
    Args:
        smi: SMILES string of the molecule
        cid: Compound ID
        pattern: SMARTS pattern for fragmentation
        
    Returns:
        Set of fragmentation strings in format: 'smi,cid,core,chains'
    """
    mol = Chem.MolFromSmiles(smi)
    outlines = set()
    
    if mol is None:
        return outlines
    
    # Fragment molecule (minCuts=2, maxCuts=2 means exactly 2 cuts)
    frags = rdMMPA.FragmentMol(mol, minCuts=2, maxCuts=2, maxCutBonds=100, 
                                pattern=pattern, resultsAsMols=False)
    
    for core, chains in frags:
        output = f'{smi},{cid},{core},{chains}'
        outlines.add(output)
    
    # If no fragmentations found, return parent molecule
    if not outlines:
        outlines.add(f'{smi},{cid},,')
    
    return outlines


def fragment_dataset(smiles: List[str], 
                     linker_min: int = 3, 
                     fragment_min: int = 5, 
                     min_path_length: int = 2, 
                     linker_leq_frags: bool = True, 
                     verbose: bool = False) -> List[List[str]]:
    """
    Fragment a dataset of SMILES strings and filter by criteria.
    
    Args:
        smiles: List of SMILES strings
        linker_min: Minimum number of heavy atoms in linker
        fragment_min: Minimum number of heavy atoms in fragments
        min_path_length: Minimum path length between fragments
        linker_leq_frags: If True, linker must be <= smallest fragment
        verbose: Print progress updates
        
    Returns:
        List of valid fragmentations, each as [cid, core, chains]
    """
    successes = []
    
    for count, smi in enumerate(smiles):
        smi = smi.rstrip()
        cmpd_id = smi
        
        # Fragment molecule
        fragmentations = fragment_mol(smi, cmpd_id)
        
        # Check if fragmentation meets criteria
        for frag_line in fragmentations:
            parts = frag_line.replace('.', ',').split(',')
            if len(parts) < 4:
                continue
                
            smiles_parts = parts[1:]  # Skip original SMILES
            mols = [Chem.MolFromSmiles(smi_part) for smi_part in smiles_parts]
            
            # Skip if any molecule failed to parse
            if None in mols:
                continue
            
            add = True
            fragment_sizes = []
            linker_size = None
            
            # Check linker (index 1)
            if len(mols) > 1 and mols[1] is not None:
                linker_size = mols[1].GetNumHeavyAtoms()
                
                # Check linker minimum size
                if linker_size < linker_min:
                    add = False
                else:
                    # Check path length between dummy atoms
                    dummy_atom_idxs = [atom.GetIdx() for atom in mols[1].GetAtoms() 
                                      if atom.GetAtomicNum() == 0]
                    if len(dummy_atom_idxs) != 2:
                        add = False
                    else:
                        path_length = len(Chem.rdmolops.GetShortestPath(
                            mols[1], dummy_atom_idxs[0], dummy_atom_idxs[1])) - 2
                        if path_length < min_path_length:
                            add = False
            else:
                add = False
            
            # Check all fragments (indices > 1) - each part is a separate fragment
            if add and len(smiles_parts) > 2:
                for i in range(2, len(smiles_parts)):
                    frag_smi = smiles_parts[i]
                    # Also handle case where fragments might be dot-separated in one part
                    frag_smiles = frag_smi.split('.')
                    
                    for frag_smi_part in frag_smiles:
                        frag_mol = Chem.MolFromSmiles(frag_smi_part)
                        if frag_mol is None:
                            add = False
                            break
                        
                        frag_size = frag_mol.GetNumHeavyAtoms()
                        fragment_sizes.append(frag_size)
                        
                        # Check each fragment minimum size
                        if frag_size < fragment_min:
                            add = False
                            break
                    
                    if not add:
                        break
                
                # Check linker <= fragments (after checking all fragments)
                if add and linker_leq_frags and linker_size is not None and fragment_sizes:
                    min_fragment_size = min(fragment_sizes)
                    if min_fragment_size < linker_size:
                        add = False
            
            if add:
                successes.append(frag_line)
        
        if verbose and count % 1000 == 0:
            print(f"\rProcessed smiles: {count}", end='')
    
    # Reformat output
    fragmentations = []
    for suc in successes:
        parts = suc.replace('.', ',').split(',')
        if len(parts) >= 4:
            fragmentations.append(parts[1:])  # Skip original SMILES
    
    return fragmentations


def get_linker(full_mol: Chem.Mol, clean_frag: Chem.Mol, starting_point: str) -> str:
    """
    Extract linker from a molecule given the starting fragments.
    
    Args:
        full_mol: Full molecule (RDKit mol object)
        clean_frag: Clean fragments without dummy atoms (RDKit mol object)
        starting_point: SMILES of starting fragments with dummy atoms
        
    Returns:
        SMILES string of the linker, or empty string if extraction fails
    """
    # Get matches of fragments
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    
    if len(matches) == 0:
        return ""
    
    # Get number of atoms in linker
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
    if linker_len == 0:
        return ""
    
    # Setup
    mol_to_break = Chem.Mol(full_mol)
    Chem.Kekulize(full_mol, clearAromaticFlags=True)
    
    poss_linker = []
    
    # Loop over matches
    for match in matches:
        mol_rw = Chem.RWMol(full_mol)
        # Get linker atoms
        linker_atoms = list(set(range(full_mol.GetNumHeavyAtoms())).difference(match))
        linker_bonds = []
        atoms_joined_to_linker = []
        
        # Get bonds between starting fragments and linker
        for idx_to_delete in sorted(match, reverse=True):
            nei = [x.GetIdx() for x in mol_rw.GetAtomWithIdx(idx_to_delete).GetNeighbors()]
            intersect = set(nei).intersection(set(linker_atoms))
            if len(intersect) == 1:
                linker_bonds.append(mol_rw.GetBondBetweenAtoms(
                    idx_to_delete, list(intersect)[0]).GetIdx())
                atoms_joined_to_linker.append(idx_to_delete)
            elif len(intersect) > 1:
                for idx_nei in list(intersect):
                    linker_bonds.append(mol_rw.GetBondBetweenAtoms(
                        idx_to_delete, idx_nei).GetIdx())
                    atoms_joined_to_linker.append(idx_to_delete)
        
        # Check number of atoms joined to linker
        if len(set(atoms_joined_to_linker)) != 2:
            continue
        
        # Delete starting fragments atoms
        for idx_to_delete in sorted(match, reverse=True):
            mol_rw.RemoveAtom(idx_to_delete)
        
        linker = Chem.Mol(mol_rw)
        
        # Check linker required num atoms
        if linker.GetNumHeavyAtoms() == linker_len:
            mol_rw = Chem.RWMol(full_mol)
            # Delete linker atoms
            for idx_to_delete in sorted(linker_atoms, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            frags = Chem.Mol(mol_rw)
            
            # Check there are two disconnected fragments
            if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                # Fragment molecule into starting fragments and linker
                fragmented_mol = Chem.FragmentOnBonds(mol_to_break, linker_bonds)
                linker_to_return = Chem.Mol(fragmented_mol)
                
                # Remove starting fragments from fragmentation
                qp = Chem.AdjustQueryParameters()
                qp.makeDummiesQueries = True
                for f in starting_point.split('.'):
                    qfrag = Chem.AdjustQueryProperties(Chem.MolFromSmiles(f), qp)
                    linker_to_return = AllChem.DeleteSubstructs(
                        linker_to_return, qfrag, onlyFrags=True)
                
                # Check linker is connected and has two bonds to outside molecule
                if len(Chem.rdmolops.GetMolFrags(linker)) == 1 and len(linker_bonds) == 2:
                    Chem.Kekulize(linker_to_return, clearAromaticFlags=True)
                    
                    # If multiple fragments, find the one matching linker length
                    if len(Chem.rdmolops.GetMolFrags(linker_to_return)) > 1:
                        for frag in Chem.MolToSmiles(linker_to_return).split('.'):
                            if Chem.MolFromSmiles(frag).GetNumHeavyAtoms() == linker_len:
                                return frag
                    
                    return Chem.MolToSmiles(Chem.MolFromSmiles(
                        Chem.MolToSmiles(linker_to_return)))
            
            # Complex cases
            else:
                fragmented_mol = Chem.MolFromSmiles(
                    Chem.MolToSmiles(fragmented_mol), sanitize=False)
                linker_to_return = AllChem.DeleteSubstructs(
                    fragmented_mol, Chem.MolFromSmiles(starting_point))
                poss_linker.append(Chem.MolToSmiles(linker_to_return))
    
    # Return results
    if len(poss_linker) == 1:
        return poss_linker[0]
    elif len(poss_linker) == 0:
        return ""
    else:
        return poss_linker[0]


def get_frags(full_mol: Chem.Mol, clean_frag: Chem.Mol, starting_point: str) -> Chem.Mol:
    """
    Extract fragments from a molecule given the linker.
    
    Args:
        full_mol: Full molecule (RDKit mol object)
        clean_frag: Clean fragments without dummy atoms (RDKit mol object)
        starting_point: SMILES of starting fragments with dummy atoms
        
    Returns:
        RDKit mol object containing the fragments
    """
    matches = list(full_mol.GetSubstructMatches(clean_frag))
    linker_len = full_mol.GetNumHeavyAtoms() - clean_frag.GetNumHeavyAtoms()
    
    if linker_len == 0:
        return full_mol
    
    Chem.Kekulize(full_mol, clearAromaticFlags=True)
    
    all_frags = []
    all_frags_lengths = []
    
    for match in matches:
        mol_rw = Chem.RWMol(full_mol)
        linker_atoms = list(set(range(full_mol.GetNumHeavyAtoms())).difference(match))
        
        for idx_to_delete in sorted(match, reverse=True):
            mol_rw.RemoveAtom(idx_to_delete)
        linker = Chem.Mol(mol_rw)
        
        if linker.GetNumHeavyAtoms() == linker_len:
            mol_rw = Chem.RWMol(full_mol)
            for idx_to_delete in sorted(linker_atoms, reverse=True):
                mol_rw.RemoveAtom(idx_to_delete)
            frags = Chem.Mol(mol_rw)
            all_frags.append(frags)
            all_frags_lengths.append(len(Chem.rdmolops.GetMolFrags(frags)))
            
            if len(Chem.rdmolops.GetMolFrags(frags)) == 2:
                return frags
    
    return all_frags[np.argmax(all_frags_lengths)] if all_frags else None

def join_fragments_linker(
    fragments_smi: str,
    linker_smi: str,
    return_mol: bool = False
) -> Optional[str]:
    """
    Join fragments and linker using atom-mapped dummy atoms.
    
    Fragments and linker have dummy atoms with atom map numbers like [*:1], [*:2].
    This function connects matching atom map numbers:
    - Fragment's [*:1] connects to Linker's [*:1]
    - Fragment's [*:2] connects to Linker's [*:2]
    
    Args:
        fragments_smi: Dot-separated SMILES of two fragments (e.g., "C([*:1])CC.N([*:2])C")
        linker_smi: SMILES of linker with attachment points (e.g., "C([*:1])C([*:2])")
        return_mol: If True, return RDKit Mol object instead of SMILES
        
    Returns:
        SMILES of joined molecule, or None if joining fails
        
    Example:
        >>> fragments = "c1ccc([*:1])cc1.c1ccc([*:2])cc1"
        >>> linker = "[*:1]CC[*:2]"
        >>> join_fragments_linker(fragments, linker)
        'c1ccc(CCc2ccccc2)cc1'
    """
    try:
        # Parse fragments
        frag_parts = fragments_smi.split('.')
        if len(frag_parts) != 2:
            return None
        
        frag1 = Chem.MolFromSmiles(frag_parts[0])
        frag2 = Chem.MolFromSmiles(frag_parts[1])
        linker = Chem.MolFromSmiles(linker_smi)
        
        if frag1 is None or frag2 is None or linker is None:
            return None
        
        # Combine all molecules
        combo = Chem.CombineMols(frag1, frag2)
        combo = Chem.CombineMols(combo, linker)
        
        # Find all dummy atoms and their attachment points, grouped by atom map number
        # Each dummy atom (atomic num 0) has a neighbor that's the real attachment point
        dummy_info = {}  # {atom_map_num: [(mol_idx, dummy_atom_idx, neighbor_idx), ...]}
        
        for atom in combo.GetAtoms():
            if atom.GetAtomicNum() == 0:  # Dummy atom
                atom_map = atom.GetAtomMapNum()
                if atom_map == 0:
                    continue  # Skip non-mapped dummies
                
                neighbors = atom.GetNeighbors()
                if len(neighbors) != 1:
                    continue
                
                neighbor_idx = neighbors[0].GetIdx()
                dummy_idx = atom.GetIdx()
                
                if atom_map not in dummy_info:
                    dummy_info[atom_map] = []
                dummy_info[atom_map].append((dummy_idx, neighbor_idx))
        
        # We need exactly 2 pairs for each atom map number (one from frag, one from linker)
        # to form bonds
        if not dummy_info:
            return None
        
        # Create editable molecule
        combo_rw = Chem.RWMol(combo)
        
        # For each atom map number, connect the two attachment points
        atoms_to_remove = set()
        for atom_map, pairs in dummy_info.items():
            if len(pairs) != 2:
                # Should have exactly 2 dummies with same map number
                continue
            
            (dummy1_idx, neighbor1_idx), (dummy2_idx, neighbor2_idx) = pairs
            
            # Add bond between the two neighbors (the real atoms)
            combo_rw.AddBond(neighbor1_idx, neighbor2_idx, Chem.rdchem.BondType.SINGLE)
            
            # Mark dummy atoms for removal
            atoms_to_remove.add(dummy1_idx)
            atoms_to_remove.add(dummy2_idx)
        
        # Remove dummy atoms (in reverse order to preserve indices)
        for atom_idx in sorted(atoms_to_remove, reverse=True):
            combo_rw.RemoveAtom(atom_idx)
        
        # Get final molecule
        final_mol = combo_rw.GetMol()
        
        # Sanitize
        try:
            Chem.SanitizeMol(final_mol)
        except Exception:
            # Try without sanitization
            pass
        
        if return_mol:
            return final_mol
        
        # Return canonical SMILES
        smiles = Chem.MolToSmiles(final_mol)
        
        # If there are multiple fragments (shouldn't happen if joining worked), take largest
        if '.' in smiles:
            smiles = sorted(smiles.split('.'), key=len, reverse=True)[0]
        
        return smiles
        
    except Exception as e:
        return None
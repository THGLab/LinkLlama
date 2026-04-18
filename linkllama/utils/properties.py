from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem import Descriptors
import numpy as np
from pathlib import Path
import useful_rdkit_utils as uru

RING_SYSTEM_FILE = Path(__file__).parent / "ring_systems" / "chembl36.csv"

UNDESIRABLE_PATTERNS = ["[C^2]1=[C^2]-[C^2]=[C^2]~[C;!d4]~[C;!^2;d2]1", "[C^2]1~[C^2]~[C^2]~[C^2]~[C;!^2;d2]~[N]1",
    "[#6^2]1~[#6^2]~[#6^3;!d4]~[#6^2]2~[#6^2]~[#6^2]~[#6^2]~[#6^2](~[*])~[#6^2]~2~[#6^2]~1",
    "[#6]1(=[*])[#6]=[#6][#6]=[#6]1", "[#6]1=[#6][R{2-}]=[R{2-}]1", "[#6^2]1~[#6^2]~[#6^2]~[#6^2]~[#6^1]~[#6^1]~1",
    "[#7,#8,#16]-[#9,#17,#35,#53]", "[r3,r4]@[r5,r6]", "[*]=[#6,#7,#8]=[*]",  # bad patterns by Eric
    "[#7,#16]=[#16]", "[#8]-[#8]",
]
PYRROLE_FORM = ["[N^2]1~[C,N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1", "[C,N;^2]1~[N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1"]
CORRECT_PYRROLE = ["[N^2]1~[C,N;^2](=[*])~[C,N;^2]~[C,N;^2]~[C;^3]1", "[N^2]1~[C,N;^2]~[C,N;^2](=[*])~[C,N;^2]~[C;^3]1",
            "[N^2]1~[C,N;^2]~[C,N;^2]~[C,N;^2](=[*])~[C;^3]1", "[C,N;^2](=[*])1~[N;^2]~[C,N;^2]~[C,N;^2]~[C;^3]1",
            "[C,N;^2]1~[N;^2]~[C,N;^2](=[*])~[C,N;^2]~[C;^3]1", "[C,N;^2]1~[N;^2]~[C,N;^2]~[C,N;^2](=[*])~[C;^3]1"]

COVALENT_WARHEADS = {
    "sulfonyl fluorides": "[#16](=[#8])(=[#8])-[#9]",
    "chloroacetamides": "[#8]=[#6](-[#6]-[#17])-[#7]",
    "cyanoacrylamides": "[#7]-[#6](=[#8])-[#6](-[#6]#[#7])=[#6]",
    "epoxides": "[#6]1-[#6]-[#8]-1",
    "aziridines": "[#6]1-[#6]-[#7]-1",
    "disulfides": "[#16]-[#16]",
    "aldehydes": "[#6](=[#8])-[#1]",
    "vinyl sulfones": "[#6]=[#6]-[#16](=[#8])(=[#8])-[#7]",
    "boronic acids/esters": "[#6]-[#5](-[#8])-[#8]",
    "acrylamides": "[#6]=[#6]-[#6](=[#8])-[#7]",
    "cyanamides": "[#6]-[#7](-[#6]#[#7])-[#6]",
    "chloroFluoroAcetamides": "[#7]-[#6](=[#8])-[#6](-[#9])-[#17]",
    "butynamides": "[#6]#[#6]-[#6](=[#8])-[#7]-[#6]",
    "chloropropionamides": "[#7]-[#6](=[#8])-[#6](-[#6])-[#17]",
    "fluorosulfates": "[#8]=[#16](=[#8])(-[#9])-[#8]",
    "beta lactams": "[#7]1-[#6]-[#6]-[#6]-1=[#8]"
}

params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_catalog = FilterCatalog.FilterCatalog(params)

def has_pains_alert(mol):
    """Check if a molecule has PAINS alert. If has, return True"""
    return PAINS_catalog.HasMatch(mol)

def has_bad_ring(mol):
    """Check if a molecule has more than 3 rings fused by one atom. If has, return True"""
    ringAtoms = []
    for ring in Chem.GetSSSR(mol):
        ringAtoms += list(ring)
    if len(ringAtoms) == 0:
        return False
    _, counts = np.unique(ringAtoms, return_counts=True)
    return counts.max() >= 3

def has_undesirable_pattern(input):
    """
    Check if a molecule has undesirable patterns. 
    Returns True if none detected. Mols with violations will have drug_likeliness score zerod.
    """
    if type(input) is str:
        mol = Chem.MolFromSmiles(input)
    else:
        mol = input
    for smarts in UNDESIRABLE_PATTERNS:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return True
    if any([mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in PYRROLE_FORM]):
        if not any([mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in CORRECT_PYRROLE]):
            return True
    return False

def check_ring_system(input, min_freq=100, ring_system_file=RING_SYSTEM_FILE):
    """
    Check if a molecule has problematic ring scaffold based on frequency.
    Returns the problematic ring scaffold string if frequency < min_freq, else ''.
    
    Args:
        input: SMILES string
        min_freq: Minimum ring frequency threshold (default: 100)
        ring_system_file: Path to ring system lookup file
    
    Returns:
        str: Problematic ring scaffold if found, '' if no problem or no rings
    Raises:
        Exception: If an error occurs during processing
    """
    try:
        # Handle different useful_rdkit_utils versions:
        # v0.3.x: RingSystemLookup.from_file(path)
        # v0.96+: RingSystemLookup(ring_file=path)
        try:
            ring_system_lookup = uru.RingSystemLookup.from_file(ring_system_file)
        except (TypeError, AttributeError):
            ring_system_lookup = uru.RingSystemLookup(ring_file=ring_system_file)
        
        mol = Chem.MolFromSmiles(input)
        ring_systems = list(ring_system_lookup.process_mol(mol))
        if len(ring_systems) == 0: return ''  # No rings is not a problem
        min_ring, min_freq_val = list(uru.get_min_ring_frequency(ring_systems))
        return min_ring if min_freq_val < min_freq else ''
    except Exception as e:
        # Raise exception instead of returning None
        raise RuntimeError(f"Error checking ring system: {e}") from e

def check_reos(input):
    """
    Check if a molecule fails REOS rules.
    Returns the failed rule description string if failed, else ''.
    
    Args:
        input: SMILES string
        check_all_rules: Unused parameter (kept for compatibility)
    
    Returns:
        str: Failed rule description if found, '' if no problem
    Raises:
        Exception: If an error occurs during processing
    """
    try:
        reos = uru.REOS()
        reos.set_active_rule_sets(["Dundee"])
        mol = Chem.MolFromSmiles(input)
        _, failed_rule = reos.process_mol(mol)
        return failed_rule if failed_rule != 'ok' else ''
    except Exception as e:
        # Raise exception instead of returning None
        raise RuntimeError(f"Error checking REOS: {e}") from e

def get_linker_topology(linker_mol):
    """
    Determine if linker is chain, branched, or contains rings.
    Returns: 'chain', 'branched', 'ring-containing', or None if linker_mol is None
    """
    if linker_mol is None:
        return None
    
    # Check for rings first
    ring_info = linker_mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        return 'ring-containing'
    
    # Check for branching (atoms with degree > 2)
    for atom in linker_mol.GetAtoms():
        if atom.GetDegree() > 2:
            return 'branched'
    
    return 'chain'

def get_linker_num_rotatable_bonds(linker_mol):
    """
    Get the number of rotatable bonds in a linker.
    Returns: int
    """
    return Descriptors.NumRotatableBonds(linker_mol)

def get_linker_num_heavy_atoms(linker_mol):
    """
    Get the number of heavy atoms in a linker.
    Returns: int
    """
    return linker_mol.GetNumHeavyAtoms()

def get_mol_mw(mol):
    """
    Get the molecular weight of a molecule.
    Returns: float
    """
    return Descriptors.MolWt(mol)

def get_mol_logp(mol):
    """
    Get the logP of a molecule.
    Returns: float
    """
    return Descriptors.MolLogP(mol)

def get_mol_tpsa(mol):
    """
    Get the tPSA of a molecule.
    Returns: float
    """
    return Descriptors.TPSA(mol)

def get_mol_num_hbd(mol):
    """
    Get the number of H-bond donors in a molecule.
    Returns: int
    """
    return Descriptors.NumHDonors(mol)

def get_mol_num_hba(mol):
    """
    Get the number of H-bond acceptors in a molecule.
    Returns: int
    """
    return Descriptors.NumHAcceptors(mol)
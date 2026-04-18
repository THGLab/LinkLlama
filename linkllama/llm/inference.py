"""
LinkLlama Inference Module

Generates linkers for molecular fragments using a fine-tuned LLM.
Supports input from CSV files or SDF fragment files.
"""

import torch
import json
import pickle
import argparse
import os
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import numpy as np
import yaml
import sys

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from linkllama.llm.constants import TEMPLATE
from linkllama.llm.sft_corpus import generate_fragment_info
from linkllama.utils.geometry import unit_vector
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem

instruction = TEMPLATE["instruction"]
input_template = TEMPLATE["input"]

# Default sampling when YAML omits keys (must match data/inference_config.yaml).
DEFAULT_TEMPERATURE = 1.4
DEFAULT_TOP_P = 0.99


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load inference configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_sampling_config(
    config: Dict[str, Any],
    num_samples_override: Optional[int] = None
) -> Dict[str, Any]:
    """
    Extract sampling configuration from config.

    temperature and top_p are read only from config['sampling'] (defaults below if null).
    num_samples can be overridden from the CLI.
    """
    sampling = config.get('sampling', {})

    temperature = sampling.get('temperature')
    top_p = sampling.get('top_p')
    num_samples = sampling.get('num_samples', 100)

    if num_samples_override is not None:
        num_samples = num_samples_override

    return {
        'num_samples': num_samples,
        'temperature': temperature if temperature is not None else DEFAULT_TEMPERATURE,
        'top_p': top_p if top_p is not None else DEFAULT_TOP_P,
        'max_length': sampling.get('max_length', 1600)
    }


# ============================================================================
# Fragment Data Extraction
# ============================================================================

def extract_fragment_info(data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Extract fragment information from input data.
    
    Accepts:
    - CSV row dict with 'fragments', 'distance_angstrom', 'angle_degrees'
    - SDF file path containing two fragments with dummy atoms
    
    Returns:
        Dictionary with 'fragments', 'distance_angstrom', 'angle_degrees'
    """
    # Handle SDF file path
    if isinstance(data, str):
        if data.endswith('.sdf'):
            sdf_data = read_sdf_fragments(data)
            if sdf_data is None:
                raise ValueError(f"Failed to read SDF file: {data}")
            return sdf_data
        else:
            raise ValueError(f"String input must be an SDF file path, got: {data}")
    
    # Handle dict (CSV row)
    if isinstance(data, dict):
        required_keys = ['fragments', 'distance_angstrom', 'angle_degrees']
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise ValueError(f"Missing required keys in data: {missing}")
        
        return {
            'fragments': str(data['fragments']),
            'distance_angstrom': float(data['distance_angstrom']),
            'angle_degrees': float(data['angle_degrees'])
        }
    
    raise ValueError(f"Unsupported data type: {type(data)}")


def read_sdf_fragments(sdf_path: str) -> Optional[Dict[str, Any]]:
    """
    Read SDF file containing two fragments with dummy atoms.
    
    Returns:
        Dictionary with 'fragments', 'distance_angstrom', 'angle_degrees' or None
    """
    try:
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
        mols = [mol for mol in supplier if mol is not None]
        
        if len(mols) != 2:
            return None
        
        frag1_mol, frag2_mol = mols[0], mols[1]
        
        # Get SMILES with dummy atoms
        frag1_smi = Chem.MolToSmiles(frag1_mol)
        frag2_smi = Chem.MolToSmiles(frag2_mol)
        fragments_smi = f"{frag1_smi}.{frag2_smi}"
        
        # Find dummy atoms and get coordinates
        dummy_coords = []
        for mol in [frag1_mol, frag2_mol]:
            if mol.GetNumConformers() == 0:
                return None
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    dummy_coords.append(np.array([pos.x, pos.y, pos.z]))
        
        if len(dummy_coords) != 2:
            return None
        
        # Calculate distance
        distance = np.linalg.norm(dummy_coords[0] - dummy_coords[1])
        
        # Calculate angle (vectors from dummy to neighbor)
        dummy_neighbors = []
        for mol in [frag1_mol, frag2_mol]:
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 0:
                    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                    if neighbors:
                        nei_pos = conf.GetAtomPosition(neighbors[0])
                        dummy_pos = conf.GetAtomPosition(atom.GetIdx())
                        vec = np.array([nei_pos.x, nei_pos.y, nei_pos.z]) - \
                              np.array([dummy_pos.x, dummy_pos.y, dummy_pos.z])
                        dummy_neighbors.append(vec)
        
        if len(dummy_neighbors) != 2:
            return None
        
        v1_u = unit_vector(dummy_neighbors[0])
        v2_u = unit_vector(dummy_neighbors[1])
        angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        return {
            'fragments': fragments_smi,
            'distance_angstrom': distance,
            'angle_degrees': angle_deg
        }
    except Exception:
        return None


# ============================================================================
# Prompt Building
# ============================================================================

def build_prompt_from_config(
    data: Union[Dict[str, Any], str],
    prompt_config: Dict[str, Any]
) -> str:
    """
    Build a prompt from data using the prompt configuration.
    
    Only fragment info is extracted from data. All property constraints
    come from prompt_config.
    """
    frag_info = extract_fragment_info(data)
    fragment_info = generate_fragment_info(
        frag_info['fragments'],
        frag_info['distance_angstrom'],
        frag_info['angle_degrees']
    )
    
    # Build linker type string
    linker_type = prompt_config.get('linker_type')
    linker_type_str = linker_type if linker_type else ""
    
    # Build linker properties string
    linker_props = []
    if rotb := prompt_config.get('rotb_range'):
        linker_props.append(f"{rotb} rotatable bonds")
    if heavy := prompt_config.get('heavy_atoms_range'):
        linker_props.append(f"{heavy} heavy atoms")
    linker_properties_str = "with " + ", ".join(linker_props) if linker_props else ""
    
    # Build molecule properties string
    mol_props = []
    if hbd := prompt_config.get('hbd_range'):
        mol_props.append(f"{hbd} H-bond donors")
    if hba := prompt_config.get('hba_range'):
        mol_props.append(f"{hba} H-bond acceptors")
    if mw := prompt_config.get('mw_range'):
        mol_props.append(f"{mw} Molecular weight")
    if logp := prompt_config.get('logp_range'):
        mol_props.append(f"{logp} LogP")
    if tpsa := prompt_config.get('tpsa_range'):
        mol_props.append(f"{tpsa} TPSA")
    # Match sft_corpus.generate_molecule_properties wording exactly
    molecule_properties_str = " And it should have the following properties: " + ", ".join(mol_props) + "." if mol_props else ""
    
    # Get reasonability (matches sft_corpus.get_reasonability_and_reasoning: "reasonable" | "unreasonable")
    reasonability = prompt_config.get('reasonability', 'reasonable')
    
    # Fill in template
    input_text = input_template
    input_text = input_text.replace('FRAGMENT_INFO', fragment_info)
    input_text = input_text.replace('LINKER_TYPE', linker_type_str)
    input_text = input_text.replace('LINKER_PROPERTIES', linker_properties_str)
    input_text = input_text.replace('MOLECULE_PROPERTIES', molecule_properties_str)
    input_text = input_text.replace('REASONABILITY', reasonability)
    input_text = ' '.join(input_text.split())  # Clean double spaces
    
    return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response: \n"


def create_prompt_fn(prompt_config: Dict[str, Any]) -> Callable:
    """Create a prompt function from config."""
    def prompt_fn(data: Union[Dict[str, Any], str]) -> str:
        return build_prompt_from_config(data, prompt_config)
    return prompt_fn


# ============================================================================
# Generation
# ============================================================================

def generate_responses(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    stopping_ids: List[int],
    num_samples: int = 100,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_length: int = 1600
) -> List[str]:
    """Generate multiple responses from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_samples,
            eos_token_id=stopping_ids,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return [tokenizer.decode(out[prompt_length:], skip_special_tokens=True).strip() 
            for out in outputs]


def _get_stopping_ids(tokenizer: AutoTokenizer) -> List[int]:
    """Get stopping token IDs for generation."""
    stopping_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id != tokenizer.unk_token_id:
            stopping_ids.append(eot_id)
    except:
        pass
    return stopping_ids


# ============================================================================
# Batch Processing
# ============================================================================

def process_batch(args):
    """Process a batch of data items on a single GPU."""
    gpu_id, model_path, data_batch, prompt_config, sampling_config = args
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if 'cuda' in device else torch.float32,
        device_map={'': device}
    )
    stopping_ids = _get_stopping_ids(tokenizer)
    prompt_fn = create_prompt_fn(prompt_config)
    
    results = {}
    for data_item in tqdm(data_batch, desc=f"GPU {gpu_id}"):
        key = data_item.get('fragments', str(data_item)) if isinstance(data_item, dict) else str(data_item)
        try:
            prompt = prompt_fn(data_item)
            responses = generate_responses(
                prompt, tokenizer, model, stopping_ids,
                num_samples=sampling_config['num_samples'],
                temperature=sampling_config['temperature'],
                top_p=sampling_config['top_p'],
                max_length=sampling_config['max_length']
            )
            
            parsed = []
            for r in responses:
                try:
                    parsed.append(json.loads(r))
                except json.JSONDecodeError:
                    parsed.append({"error": "json_format_error", "raw": r})
            
            results[key] = {'prompt': prompt, 'responses': parsed, 'data': data_item}
        except Exception as e:
            results[key] = {'error': str(e), 'data': data_item}
    
    return results


# ============================================================================
# Main Inference Functions
# ============================================================================

def run_inference(
    model_path: str,
    data_items: List[Union[Dict[str, Any], str]],
    config: Dict[str, Any],
    save_path: Optional[str] = None,
    num_gpus: Optional[int] = None,
    num_samples_override: Optional[int] = None
) -> Dict[Any, Any]:
    """
    Run inference on a list of data items.

    temperature / top_p come only from config['sampling'] (see DEFAULT_TEMPERATURE / DEFAULT_TOP_P if null).
    num_samples_override: optional CLI override for num_samples.
    """
    prompt_config = config.get('prompt', {})
    sampling_config = get_sampling_config(config, num_samples_override)
    
    num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
    num_gpus = max(1, num_gpus)
    print(f"GPUs: {num_gpus}, Sampling: temp={sampling_config['temperature']}, "
          f"top_p={sampling_config['top_p']}, n={sampling_config['num_samples']}")
    
    if num_gpus > 1:
        pool = mp.Pool(num_gpus)
        try:
            batches = [data_items[i::num_gpus] for i in range(num_gpus)]
            args_list = [(i, model_path, batch, prompt_config, sampling_config) 
                         for i, batch in enumerate(batches)]
            results_list = pool.map(process_batch, args_list)
            combined_results = {}
            for r in results_list:
                combined_results.update(r)
        finally:
            pool.close()
            pool.join()
    else:
        combined_results = process_batch((0, model_path, data_items, prompt_config, sampling_config))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(combined_results, f)
        print(f"Results saved to: {save_path}")
    
    return combined_results


def run_single_inference(
    data: Union[Dict[str, Any], str],
    model_path: str,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    gpu_id: int = 0
) -> Dict[str, Any]:
    """
    Run inference on a single input.

    temperature / top_p are taken only from config YAML sampling section.
    num_samples: optional override for num_samples.
    """
    if config is None:
        if config_path:
            config = load_config(config_path)
        else:
            config = {'prompt': {'reasonability': 'reasonable'}, 'sampling': {}}
    
    prompt_config = config.get('prompt', {})
    sampling_config = get_sampling_config(config, num_samples)
    
    frag_info = extract_fragment_info(data)
    prompt = build_prompt_from_config(frag_info, prompt_config)
    
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16 if 'cuda' in device else torch.float32,
        device_map={'': device}
    )
    
    responses = generate_responses(
        prompt, tokenizer, model, _get_stopping_ids(tokenizer),
        num_samples=sampling_config['num_samples'],
        temperature=sampling_config['temperature'],
        top_p=sampling_config['top_p'],
        max_length=sampling_config['max_length']
    )
    
    parsed = []
    for r in responses:
        try:
            parsed.append(json.loads(r))
        except json.JSONDecodeError:
            parsed.append({"error": "json_format_error", "raw": r})
    
    return {'prompt': prompt, 'responses': parsed, 'fragment_info': frag_info}


# ============================================================================
# Output Path Resolution
# ============================================================================

def resolve_output_path(
    input_path: str,
    config: Dict[str, Any],
    cli_output: Optional[str] = None,
    cli_suffix: Optional[str] = None
) -> str:
    """
    Resolve output path from CLI override, config, or auto-generate.
    
    Priority: 
      - CLI --output (exact path)
      - config output_path (exact path)
      - auto-generate: input_stem + suffix + .pkl
        - suffix priority: CLI --suffix > config output_suffix > "_results"
        - output_dir: config output_dir or input directory
    """
    # CLI --output takes highest priority (exact path)
    if cli_output:
        return cli_output
    
    output_config = config.get('output', {})
    
    # Config output_path takes second priority (exact path)
    if output_config.get('output_path'):
        return output_config['output_path']
    
    # Auto-generate from input path
    input_p = Path(input_path)
    
    # Suffix priority: CLI > config > default
    suffix = cli_suffix or output_config.get('output_suffix', '_results')
    output_dir = output_config.get('output_dir')
    
    # Build output filename
    if input_p.is_dir():
        output_name = f"{input_p.name}{suffix}.pkl"
    else:
        output_name = f"{input_p.stem}{suffix}.pkl"
    
    # Determine output directory
    if output_dir:
        return str(Path(output_dir) / output_name)
    elif input_p.is_dir():
        return str(input_p / output_name)
    else:
        return str(input_p.parent / output_name)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LinkLlama inference for linker generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (model_path from config)
  python inference.py --config config.yaml --csv data.csv

  # Override num_samples and filename suffix
  python inference.py --config config.yaml --csv data.csv --num_samples 50 --suffix "_n50"

  # Specify exact output file
  python inference.py --config config.yaml --csv data.csv --output results/exp1.pkl
        """
    )
    
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--csv", type=str, help="Path to CSV file with fragments data")
    parser.add_argument("--sdf_dir", type=str, help="Path to directory containing SDF files")
    parser.add_argument("--model", type=str, help="Path to the model (overrides config)")
    parser.add_argument("--output", type=str, help="Exact output file path")
    parser.add_argument("--suffix", type=str, help="Output filename suffix (e.g., '_t1.0_p0.7')")
    parser.add_argument("--gpus", type=int, help="Number of GPUs (default: all available)")
    
    parser.add_argument("--num_samples", type=int, help="Override num_samples from config")
    
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)
    
    # Load config
    config = load_config(args.config)
    
    # Get model path (CLI override > config)
    model_path = args.model or config.get('sampling', {}).get('model_path')
    if not model_path:
        parser.error("Model path must be specified in config (sampling.model_path) or via --model")
    print(f"Model: {model_path}")
    
    # Load data
    if args.csv:
        import pandas as pd
        print(f"Loading CSV: {args.csv}")
        df = pd.read_csv(args.csv)
        data_items = df.to_dict('records')
        print(f"Loaded {len(data_items)} rows")
        input_path = args.csv
    elif args.sdf_dir:
        print(f"Loading SDFs from: {args.sdf_dir}")
        sdf_dir = Path(args.sdf_dir)
        data_items = [str(f) for f in sdf_dir.glob("*.sdf")]
        print(f"Found {len(data_items)} SDF files")
        input_path = args.sdf_dir
    else:
        parser.error("Must provide either --csv or --sdf_dir")
    
    # Resolve output path
    save_path = resolve_output_path(input_path, config, args.output, args.suffix)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    print(f"Output: {save_path}")
    
    # Run inference
    results = run_inference(
        model_path=model_path,
        data_items=data_items,
        config=config,
        save_path=save_path,
        num_gpus=args.gpus,
        num_samples_override=args.num_samples
    )
    
    print(f"Processed {len(results)} items")

import logging
import shutil
from pathlib import Path
import numpy as np
from chai_lab.chai1 import run_inference
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import torch
import os
import hashlib
import time
import random
from typing import List, Union

def generate_unique_hash():
    """
    Generate a unique hash ID based on timestamp, process ID, and random number.
    """
    timestamp = str(time.time())
    process_id = str(os.getpid())
    random_num = random.randint(0, 1000000)
    
    data_to_hash = f"{timestamp}-{process_id}-{random_num}"
    
    return hashlib.sha256(data_to_hash.encode()).hexdigest()[:16]

def convert_cif_paths_to_pdb(cif_paths, seed):
    """
    Convert a list of CIF file paths to PDB file paths by parsing and saving the
    structures with the PDB module.
    
    Parameters
    ----------
    cif_paths : List[str]
        A list of paths to CIF files.
    seed : int
        Random seed used to generate a unique ID for the output pdb file
    
    Returns
    -------
    List[str]
        A list of paths to the output PDB files.
    """
    pdb_paths = []
    parser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()
    
    for cif_path in cif_paths:
        # Create output PDB path by changing extension
        pdb_path = str(Path(cif_path).with_suffix('.pdb'))
        # add seed info to file name
        pdb_path = pdb_path.replace('.pdb', f'_{seed}.pdb')
        
        # Parse and save structure
        structure = parser.get_structure('structure', cif_path)
        io.set_structure(structure)
        io.save(pdb_path)
        
        pdb_paths.append(pdb_path)
    
    return pdb_paths
    

def predict(fasta_path: Path, output_dir: Path, seed: int = 0, use_msa: bool = False): 
    """
    Run Chai-1 inference on a given protein sequence.

    Parameters
    ----------
    fasta_path : Path
        Path to a FASTA file containing the protein sequence.
    output_dir : Path
        Directory where the output files will be written.
    seed : int, optional
        Random seed used to generate a unique ID for the output files.
        Defaults to 0.
    use_msa : bool, optional
        Whether to use a multiple sequence alignment (MSA) for inference.
        Defaults to False.

    Returns
    -------
    structure_path : Path
        Path to the generated structure file (PDB format).
    scores : List[float]
        List of scores for the generated structures.
    """
    fasta_name = fasta_path.stem
    hash_id = generate_unique_hash()
    # create tmp output directory. Inference expects an empty directory; enforce this
    tmp_output_dir = Path(os.path.join(output_dir, f'tmp/{hash_id}'))
    tmp_output_dir.mkdir(parents=True, exist_ok=False)
    # run chai 
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=tmp_output_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        device="cuda:0",
        use_esm_embeddings=False if use_msa else True, 
        seed=seed,
        use_msa_server=use_msa
    )
    #convert cifs to pdbs for compatibility
    cif_paths = candidates.cif_paths
    pdb_paths = convert_cif_paths_to_pdb(cif_paths, hash_id)
    
    #aggregate scores across structures
    agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

    # Load pTM, ipTM, pLDDTs and clash scores for sample the best sample
    best_sample = agg_scores.index(max(agg_scores))
    scores = np.load(tmp_output_dir.joinpath(f"scores.model_idx_{best_sample}.npz"))
    pae_matrix, pae, plddt = candidates.pae[best_sample], candidates.pae[best_sample], candidates.plddt[best_sample]
    scores_dict = {key: scores[key] for key in scores.keys()}
    scores_dict['pae_matrix'] = pae_matrix
    scores_dict['aggregate_score'] = scores['aggregate_score'][0]   
    scores_dict['pae'] = torch.mean(pae).item()
    scores_dict['plddt'] = torch.mean(plddt).item()
    scores_dict['ptm'] = scores_dict['ptm'][0]
    scores_dict['iptm'] = scores_dict['iptm'][0]

    # return best pdb path
    pdb_path = pdb_paths[best_sample]
    output_path = os.path.join(output_dir, fasta_name + '_chai.pdb')

    shutil.copy(pdb_path, output_path)

    shutil.rmtree(tmp_output_dir, ignore_errors=True)

    return output_path, scores_dict



if __name__ == "__main__":
    structure_path, scores = predict(fasta_path=Path('examples/5GNJ.fasta'), output_dir=Path('output'))
    print(f"Structure saved to {structure_path}")
    print(f"Confidence Metrics Available: {list(scores.keys())}")
    for metric in ['aggregate_score', 'pae', 'plddt', 'ptm', 'iptm']:
        print(f"{metric}: {scores[metric]:.3f}")

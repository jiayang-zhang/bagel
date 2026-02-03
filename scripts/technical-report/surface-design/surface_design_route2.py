import argparse
from configs.configs import get_cfg_defaults

import copy
import logging
import numpy as np
import pathlib as pl
import os
from datetime import datetime

from biotite.structure import AtomArray
import biotite.structure.io.pdb as pdb
from biotite.database import rcsb

import sys
sys.path.append('/')
import bagel as bg
from bagel.energies import TemplateMatchEnergy
from bagel.constants import aa_dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def arg_parse():
    parser = argparse.ArgumentParser(description='Surface design route 2')
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--outdir", type=str, required=True, help="path to output folder")

    # -------------------------
    # Energy term params
    # -------------------------
 
    parser.add_argument(
        "--tme_template_distogram",
        type=str2bool,
        required=True,
        help="TemplateMatchEnergy to the Template residue: True=distogram, False=RMSD"
    )
    
    parser.add_argument(
        "--tme_template_weight",
        type=float,
        required=True,
        help="TemplateMatchEnergy to the Template residue: weight"
    )

    parser.add_argument(
        "--tme_base_distogram",
        type=str2bool,
        required=True,
        help="TemplateMatchEnergy to the base scaffold: True=distogram, False=RMSD"
    )

    parser.add_argument(
        "--tme_base_weight",
        type=float,
        required=True,
        help="TemplateMatchEnergy to the base scaffold: weight"
    )

    parser.add_argument(
        "--ptm_weight",
        type=float,
        required=True,
        help="PTM energy weight"
    )

    parser.add_argument(
        "--overall_plddt_weight",
        type=float,
        required=True,
        help="overall pLDDT weight"
    )
    

    args = parser.parse_args()
    return args




def main():
    
    # ---- setup config ----
    cfg = get_cfg_defaults()
    args = arg_parse()
    cfg.merge_from_file(args.config)
    
    use_modal = cfg.SETUP.USE_MODAL
    
    output_dir = args.outdir

    # PART 1: Define the target protein  
    # COLICIN E7 IMMUNITY PROTEIN
    # UniProt ID: Q03708, PDB ID: 1UNK
    base_sequence = 'MELKNSISDYTEAEFVQLLKEIEKENVAATDDVLDVLLEHFVKITEHPDGTDLIYYPSDNRDDSPEGIVKEIKEWRAANGKPGFKQG'

    dummy_holder = []
    mutation_indexes, res_before_mutation, res_after_mutation = [], [], []
    res_optim_idexes, res_optim = [], []
    base_sequence_after_mutation = list(copy.deepcopy(base_sequence))

    # 0-indexed list
    for id, AA in enumerate(base_sequence):
        if AA == 'K':
            dummy_holder.append(AA)
            if len(dummy_holder) <=5:
                res_optim_idexes.append(id)
                res_optim.append(AA)
            else:
                mutation_indexes.append(id)
                res_before_mutation.append(AA)
                # This is because at physiological pH (about 7.4), Arginine (R) is positively charged like Lysine (K)
                # Not using Histidine (H) because its charge state is pH-dependent and less reliably positive at physiological pH
                res_after_mutation.append('R')
                base_sequence_after_mutation[id] = 'R'
    base_sequence_after_mutation = ''.join(base_sequence_after_mutation)


    # All the residues in the sequence can be optimised to obtain the desired structure
    # but the AAs cannot be changed to K, the number of K is conserved
    mutability = [
        False if i in res_optim_idexes else True
        for i in range(len(base_sequence_after_mutation))
    ]

    base_residues = [
        bg.Residue(name=aa, chain_ID='A', index=i, mutable=mut) # 0-indexed as well
        for i, (aa, mut) in enumerate(zip(base_sequence_after_mutation, mutability))
    ]

    base_chain = bg.Chain(residues=base_residues)

    # Those left-out Lysicines in base sequence for comparision with the template
    base_residues_for_compare = [base_residues[i] for i in res_optim_idexes]


    # PART 2: Define the template protein

    # ======= Define a perfect CA-only template structure =======
    N = len(res_optim_idexes)
    template = AtomArray(N)

    # TODO: Define the spacing between Ca atoms
    template.coord = np.array([[10.0 * i, 0.0, 0.0] for i in range(N)])
    template.atom_name = np.array(["CA"] * N) # Alpha Carbon
    template.element = np.array(["C"] * N) # Carbon for CA
    template.chain_id = np.array(["A"] * N) # Single chain A
    template.res_id = np.array([i for i in range(N)])
    template.res_name = np.array([bg.constants.aa_dict[res] for res in res_optim]) # Three letter for residue names to satisfy AtomArray format


    # ======= Extract the orginal structure of COLICIN E7 IMMUNITY PROTEIN | UniProt ID: Q03708, PDB ID | a CA-only template structure  =======
    
    pdb_path = rcsb.fetch("1UNK", "pdb")
    pdb_file = pdb.PDBFile.read(pdb_path)
    # Get the AtomArray
    structure = pdb.get_structure(pdb_file, model=1)   
    base_sequence_template = structure[(structure.chain_id == "A") & (structure.atom_name == "CA")]



    # PART 3: Define the Oracles and EnergyTerms
    # ======= Define the ESMFold Oracle ======= 
    config = {
        'output_pdb': False,
        'output_cif': False,
    }

    esmfold = bg.oracles.ESMFold(
        use_modal=use_modal, config=config
    )

    energy_terms = [
        
        # Of the target template structure
        TemplateMatchEnergy(
            oracle=esmfold,
            template_atoms=template,
            residues=base_residues_for_compare, # the selected lyscines in base sequence
            backbone_only=True,  # Make the inputs Ca-only
            distogram_separation=args.tme_template_distogram,  # use distogram separation to calculate
            weight=args.tme_template_weight,
            name = 'TME_template'
        ),

        # Of the base sequence structure
        TemplateMatchEnergy(
            oracle=esmfold,
            template_atoms=base_sequence_template,
            residues=base_residues, # the original base sequence 
            backbone_only=True,  # Make the inputs Ca-only
            distogram_separation=args.tme_base_distogram,  # use distogram separation to calculate
            weight=args.tme_base_weight,
            name = 'TME_base'
        ),

        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=args.ptm_weight,
        ),

        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=args.overall_plddt_weight,
        ),
    ]

    # Define the state
    state = bg.State(
        name='state_A',
        chains=[base_chain],
        energy_terms=energy_terms,
    )

    initial_system = bg.System(states=[state])

    mutation_bias_no_cystein_no_lysine = {aa: 1.0 / (len(aa_dict) - 2) if (aa != 'C' and aa != 'K') else 0.0 for aa in
                                          aa_dict.keys()}
    mutator = bg.mutation.GrandCanonical(
        mutation_bias= mutation_bias_no_cystein_no_lysine,
        move_probabilities = {
            'substitution': cfg.MOVE.SUB,
            'addition': cfg.MOVE.ADD,
            'removal': cfg.MOVE.REMOVE,
            'swap': cfg.MOVE.SWAP,
        }
    )

    optimization_params = {
        'high_temperature': cfg.OPTIM.HIGH_TEMP,
        'low_temperature': cfg.OPTIM.LOW_TEMP,
        'n_steps_high': cfg.OPTIM.N_STEPS_HIGH,
        'n_steps_low': cfg.OPTIM.N_STEPS_LOW,
        'n_cycles': cfg.OPTIM.N_CYCLES,
    }


    current_dir = os.path.dirname(os.path.abspath(__file__))
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=mutator,
        high_temperature=optimization_params['high_temperature'],
        low_temperature=optimization_params['low_temperature'],
        n_steps_high=optimization_params['n_steps_high'],
        n_steps_low=optimization_params['n_steps_low'],
        n_cycles=optimization_params['n_cycles'],
        preserve_best_system_every_n_steps=optimization_params['n_steps_high'] + optimization_params['n_steps_low'],
        log_frequency=1,
        log_path=pl.Path(os.path.join(current_dir, output_dir)),
    )

    # Return the best system
    best_system = minimizer.minimize_system(system=initial_system)


if __name__ == '__main__':
    main()
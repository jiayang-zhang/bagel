import argparse
from configs.configs import get_cfg_defaults

import copy
import logging
import numpy as np
import pathlib as pl
import os
from datetime import datetime

from biotite.structure import AtomArray

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
    parser = argparse.ArgumentParser(description='Surface design route 1')
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--outdir", type=str, required=True, help="path to output folder")

    # -------------------------
    # Energy term params
    # -------------------------
 
    parser.add_argument(
        "--tme_distogram",
        type=str2bool,
        required=True,
        help="TemplateMatchEnergy: True=distogram, False=RMSD"
    )
    
    parser.add_argument(
        "--tme_weight",
        type=float,
        required=True,
        help="TemplateMatchEnergy weight"
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
    # UniProt ID: P42212
    base_sequence = 'KKKKK'

    # All the residues in the sequence are immutable
    # and the AAs cannot be changed to K, the number of K is conserved
    # Set it mutable becasue bagel cannot work with zero devision for mutation.
    mutability = [False for i in range(len(base_sequence))]

    base_residues = [
        bg.Residue(name=aa, chain_ID='A', index=i, mutable=mut) # 0-indexed as well
        for i, (aa, mut) in enumerate(zip(base_sequence, mutability))
    ]

    base_chain = bg.Chain(residues=base_residues)
    
    

    # PART 2: Define the template protein

    # ======= Define a perfect CA-only template structure =======
    # Starting with 5 atoms
    N = len(base_residues)
    template = AtomArray(N)
    template.coord = np.array([[10.0 * i, 0.0, 0.0] for i in range(N)])
    template.atom_name = np.array(["CA"] * N) # Alpha Carbon
    template.element = np.array(["C"] * N) # Carbon for CA
    template.chain_id = np.array(["A"] * N) # Single chain A
    template.res_id = np.array([i for i in range(N)])
    template.res_name = np.array([bg.constants.aa_dict[res] for res in base_sequence]) # Three letter for residue names to satisfy AtomArray format



    # PART 3: Define the Oracles and EnergyTerms

    # Define the ESMFold Oracle
    esmfold_config = {
        'output_pdb': False,
        'output_cif': False,
    }

    esmfold = bg.oracles.ESMFold(
        use_modal=use_modal, config=esmfold_config
    )

    energy_terms = [
        TemplateMatchEnergy(
            oracle=esmfold,
            template_atoms=template,
            residues=base_residues, 
            backbone_only=True,  # Make the inputs Ca-only
            distogram_separation=args.tme_distogram,  # use distogram separation to calculate
            weight=args.tme_weight,
        ),

        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=args.ptm_weight,
        ),

        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=args.overall_plddt_weight,
        ),

        # bg.energies.SecondaryStructureEnergy(
        #     oracle=esmfold,
        #     residues=base_residues,
        #     target_secondary_structure='alpha-helix',
        #     weight=20,
        # )
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
    log_dir = pl.Path(args.outdir)
    if not log_dir.is_absolute():
        log_dir = pl.Path(current_dir) / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    minimizer = bg.minimizer.SimulatedTempering(
        mutator=mutator,
        high_temperature=optimization_params['high_temperature'],
        low_temperature=optimization_params['low_temperature'],
        n_steps_high=optimization_params['n_steps_high'],
        n_steps_low=optimization_params['n_steps_low'],
        n_cycles=optimization_params['n_cycles'],
        preserve_best_system_every_n_steps=optimization_params['n_steps_high'] + optimization_params['n_steps_low'],
        log_frequency=1,
        log_path=log_dir,
    )


    # Return the best system
    best_system = minimizer.minimize_system(system=initial_system)


if __name__ == '__main__':
    main()
import copy
import logging
import numpy as np
import pathlib as pl
import os
from datetime import datetime

from biotite.structure import AtomArray
from biotite.database.rcsb import fetch

import sys
sys.path.append('/')
import bagel as bg
from bagel.energies import TemplateMatchEnergy
from bagel.constants import aa_dict
from bagel.utils import get_atomarray_in_residue_range, get_sequence_from_pdb_id, get_reconciled_sequence, sequence_from_atomarray, get_atomarray_by_residue_ids



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def main():
    use_modal = False
    optimization_params = None
    output_dir = 'data/lysozyme-route1'

    # PART 1: Define the target protein
    # glu35 and asp 52
    base_sequence = 'ED'

    # All the residues in the sequence are mutable
    mutability = [False for i in range(len(base_sequence))]

    base_residues = [
        bg.Residue(name=aa, chain_ID='A', index=i, mutable=mut) # 0-indexed as well
        for i, (aa, mut) in enumerate(zip(base_sequence, mutability))
    ]

    base_chain = bg.Chain(residues=base_residues)
    
    

    # PART 2: Define the template protein
    # ======= Define a template structure =======
    chain_atoms = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1LYZ", format="pdb"))
    active_site_atoms = get_atomarray_by_residue_ids(
        chain_atoms,
        res_ids=[35, 52],
        chain="A"
    )
    template = active_site_atoms



    # PART 3: Define the Oracles and EnergyTerms

    # Define the ESMFold Oracle
    config = {
        'output_pdb': False,
        'output_cif': False,
    }

    esmfold = bg.oracles.ESMFold(
        use_modal=use_modal, config=config
    )

    energy_terms = [
        TemplateMatchEnergy(
            oracle=esmfold,
            template_atoms=template,
            residues=base_residues, 
            backbone_only=False,
            distogram_separation=True,  # use distogram separation to calculate
            weight=1,
        ),

        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1,
        ),

        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=1,
        ),

        # This term tries to minimize hydrophobic residues exposed at the surface            
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            surface_only=True,
            weight=float(len(base_sequence)),
            name="hydrophilic_surface",
        ),
        # This term tries to MAXIMISE (weight is negative!) hydrophobic residues inside the core
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            core_only=True,
            weight=-float(len(base_sequence)),
            name="hydrophobic_core",
        ),

        # Total exposed surface area (SASA) and have neagtive weight to make the overall less compact
        bg.energies.SurfaceAreaEnergy(
            oracle=esmfold,
            residues=base_residues,  
            weight= -1,
            inheritable=False, 
            name='SASA'
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


    mutator = bg.mutation.GrandCanonical(
        move_probabilities = {
            'substitution': 0.25,
            'addition': 0.25,
            'removal': 0.25,
            'swap': 0.25,
        }
    )

    if optimization_params is None:
        optimization_params = {
            'high_temperature': 1,
            'low_temperature': 0.05,
            'n_steps_high': 100,
            'n_steps_low': 400,
            'n_cycles': 10000,
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
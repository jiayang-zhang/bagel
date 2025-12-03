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


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)





def main():
    use_modal = False
    optimization_params = None
    output_dir = 'data/surface-structure-optim-route2'

    # PART 1: Define the target protein
    # UniProt ID: P42212
    base_sequence = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

    dummy_holder = []
    mutation_indexes, res_before_mutation, res_after_mutation = [], [], []
    res_optim_idexes, res_optim = [], []
    base_sequence_after_mutation = list(copy.deepcopy(base_sequence))

    # 0-indexed list
    for id, AA in enumerate(base_sequence):
        if AA == 'K':
            dummy_holder.append(AA)
            if len(dummy_holder) <=2:
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
    mutability = [True for i in range(len(base_sequence_after_mutation))]

    base_residues = [
        bg.Residue(name=aa, chain_ID='A', index=i, mutable=mut) # 0-indexed as well
        for i, (aa, mut) in enumerate(zip(base_sequence_after_mutation, mutability))
    ]

    base_chain = bg.Chain(residues=base_residues)

    base_residues_for_compare = [base_residues[i] for i in res_optim_idexes]


    # PART 2: Define the template protein

    # ======= Define a perfect CA-only template structure =======
    # Starting with 5 atoms
    N = len(res_optim_idexes)
    template = AtomArray(N)

    # TODO: Define the spacing between Ca atoms
    template.coord = np.array([[10.0 * i, 0.0, 0.0] for i in range(N)])
    template.atom_name = np.array(["CA"] * N) # Alpha Carbon
    template.element = np.array(["C"] * N) # Carbon for CA
    template.chain_id = np.array(["A"] * N) # Single chain A
    template.res_id = np.array([0, 1]) # TODO: Check if this [0,1] should be the same as 'res_optim_idexes' [2,25]
    template.res_name = np.array([bg.constants.aa_dict[res] for res in res_optim]) # Three letter for residue names to satisfy AtomArray format



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
            residues=base_residues_for_compare,
            backbone_only=True,  # Make the inputs Ca-only
            distogram_separation=True,  # use distogram separation to calculate
            weight=10.0,
        ),

        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1.0,
        ),

        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=1.0,
        ),
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
            'substitution': 0,
            'addition': 0,
            'removal': 0,
            'swap': 1.0,  # Only allow swap moves
        }
    )

    if optimization_params is None:
        optimization_params = {
            'high_temperature': 1.0,
            'low_temperature': 0.1,
            'n_steps_high': 100,
            'n_steps_low': 400,
            'n_cycles': 100,
        }


    current_dir = os.path.dirname(os.path.abspath(__file__))
    # current_dir = os.getcwd() # for jupyter
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
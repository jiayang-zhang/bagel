import copy
import logging
import numpy as np
import pathlib as pl
import os
from datetime import datetime

from biotite.structure import AtomArray
import biotite.structure.io.pdb as pdb
from biotite.database import rcsb
from biotite.database.rcsb import fetch
from biotite.structure.filter import filter_amino_acids

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
    output_dir = 'data/lysozyme-route2'

    # PART 1: Define the target protein  
    # COLICIN E7 IMMUNITY PROTEIN
    # UniProt ID: Q03708, PDB ID: 1UNK
    base_sequence = 'MELKNSISDYTEAEFVQLLKEIEKENVAATDDVLDVLLEHFVKITEHPDGTDLIYYPSDNRDDSPEGIVKEIKEWRAANGKPGFKQG'
    
    # Picked them out according to the CA-CA distance most similar to that of lysozyme's active site's
    idx_E = 22  
    idx_D = 62  
    

    # Except idx_E and idx_D, all the residues in the sequence can be optimised to obtain the desired structure
    mutability = [False if i in [idx_E, idx_D] else True for i in range(len(base_sequence))]

    base_residues = [
        bg.Residue(name=aa, chain_ID='A', index=i, mutable=mut) # 0-indexed as well
        for i, (aa, mut) in enumerate(zip(base_sequence, mutability))
    ]

    base_chain = bg.Chain(residues=base_residues)

    # For TemplateMatching with the active site
    base_residues_for_compare = [base_residues[i] for i in (idx_E, idx_D)]


    # PART 2: Define the template protein
    # Active site of Hen egg‑white lysozyme | PDB ID：1LYZ
    # ======= Define a backbone +  side chain heavy atom template structure =======
    
    lysozyme_atoms = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1LYZ", format="pdb"))
    active_site_atoms = get_atomarray_by_residue_ids(
        lysozyme_atoms,
        res_ids=[35, 52],
        chain="A"
    )
    template = active_site_atoms

    

    # ======= Extract the orginal structure of COLICIN E7 IMMUNITY PROTEIN | UniProt ID: Q03708, PDB ID | a CA-only template structure  =======
    # immunity_protein = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1UNK", format="pdb"))
    # immunity_protein = immunity_protein[filter_amino_acids(immunity_protein)]  # AA only，removing HOH
    # immunity_protein = immunity_protein[immunity_protein.element != "H"] #remove all H atoms
    # end_res = immunity_protein.res_id.max()
    # base_sequence_template = get_atomarray_in_residue_range(atoms=immunity_protein, start=0, end=end_res, chain="A")

    base_scaffold = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1UNK", format="pdb"))
    base_scaffold_AAatoms = base_scaffold[filter_amino_acids(base_scaffold)]          # AA only
    base_scaffold_AAatoms = base_scaffold_AAatoms[base_scaffold_AAatoms.chain_id == "A"] # chain A only
    
    # remove hydrogen atoms
    if "element" in base_scaffold_AAatoms.get_annotation_categories():
        base_scaffold_AAatoms = base_scaffold_AAatoms[base_scaffold_AAatoms.element != "H"]
    else:
        base_scaffold_AAatoms = base_scaffold_AAatoms[~np.char.startswith(base_scaffold_AAatoms.atom_name.astype(str), "H")]
    
    start_res = base_scaffold_AAatoms.res_id.min()
    end_res = base_scaffold_AAatoms.res_id.max()
    base_scaffold_template = get_atomarray_in_residue_range(atoms=base_scaffold_AAatoms, start=start_res, end=end_res, chain="A")




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
            residues=base_residues_for_compare, 
            backbone_only=False, 
            distogram_separation=True,  # use distogram separation to calculate
            weight=10,
            name = 'TME_template'
        ),

        # Of the base sequence structure
        TemplateMatchEnergy(
            oracle=esmfold,
            template_atoms=base_scaffold_template,
            residues=base_residues, # the original base sequence 
            backbone_only=True,  # Make the inputs Ca-only
            distogram_separation=True,  # use distogram separation to calculate
            weight=1,
            name = 'TME_base'
        ),

        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=1,
        ),

        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=1,
        ),

        # let surface be hydrophilic (can also have another term to define the core to be hydrophobic)
        bg.energies.HydrophobicEnergy(
            oracle=esmfold,
            mode='surface', 
            weight= -1,
            
        ),

        # Active site hydrophilic term (negative for hydrophilic surface)
        bg.energies.SurfaceAreaEnergy(
            oracle=esmfold,
            residues=base_residues,  
            weight= -1,
            inheritable=False, 
            name='SASA'
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
            'substitution': 0.5,
            'addition': 0.0,
            'removal': 0.0,
            'swap': 0.5,
        }
    )

    if optimization_params is None:
        optimization_params = {
            'high_temperature': 0.5,
            'low_temperature': 0.05,
            'n_steps_high': 100,
            'n_steps_low': 400,
            'n_cycles': 10000,
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
import copy
import logging
import numpy as np
import pathlib as pl
import os
from datetime import datetime

from biotite.structure import AtomArray
from biotite.database.rcsb import fetch
from biotite.structure.filter import filter_amino_acids

import sys
sys.path.append('/')
import bagel as bg
from bagel.energies import TemplateMatchEnergy, SurfaceAreaEnergy
from bagel.constants import aa_dict
from bagel.utils import get_atomarray_in_residue_range, get_sequence_from_pdb_id, get_reconciled_sequence, sequence_from_atomarray, get_atomarray_by_residue_ids


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



def main():
    use_modal = False
    optimization_params = None
    output_dir = 'data/hydrolase-route1'

    # PART 1: Define the target protein
    # PDB: 1UA7
    base_sequence = 'PSIKSGTILHAWNWSFNTLKHNMKDIHDAGYTAIQTSPINQVKEGNQGDKSMSNWYWLYQPTSYQIGNRYLGTEQEFKEMCAAAEEYGIKVIVDAVINHTTFDYAAISNEVKSIPNWTHGNTQIKNWSDRWDVTQNSLLGLYDWNTQNTQVQSYLKRFLERALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITNTSAEFQYGEILQDSASRDAAYANYMDVTASNYGHSIRSALKNRNLGVSNISHYASDVSADKLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGKSQIGDRGSALFEDQAITAVNRFHNVMAGQPEELSNPQGNNQIFMNQRGSHGVVLANAGSSSVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYPD'
    # base_sequence = get_sequence_from_pdb_id(pdb_id="1UA7", sequence_index=0)


    # Continuous numbering in python (indexing from 0)
    # All the residues in the sequence are mutable
    natural_active_site = [172, 204, 265]
    # TODO: Picked them out according to the CA-CA distance most similar to that of active site's
    artificial_active_site = [328, 347, 387]
    

    # Except idx_E and idx_D, all the residues in the sequence can be optimised to obtain the desired structure
    mutability = [False if i in [natural_active_site, artificial_active_site] else True for i in range(len(base_sequence))]


    base_residues = [
        bg.Residue(name=aa, chain_ID='A', index=i, mutable=mut) # 0-indexed as well
        for i, (aa, mut) in enumerate(zip(base_sequence, mutability))
    ]

    base_chain = bg.Chain(residues=base_residues)

    activesite1 = [base_residues[i] for i in natural_active_site]
    activesite2 = [base_residues[i] for i in artificial_active_site]


    
    # PART 2: Define the template protein
    # ======= Define a perfect CA-only template structure =======

    # NOT continuous numbering in PDB (indexing from 5)
    chain_atoms = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1UA7", format="pdb"))
    active_site_atoms = get_atomarray_by_residue_ids(
        chain_atoms,
        res_ids=[176, 208, 269],
        chain="A"
    )
    template = active_site_atoms


    chain_atoms2 = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1UA7", format="pdb"))
    active_site_atoms2 = get_atomarray_by_residue_ids(
        chain_atoms2,
        res_ids=[176, 208, 269],
        chain="A"
    )
    template2 = active_site_atoms2

    
    base_scaffold = bg.oracles.folding.utils.pdb_file_to_atomarray(fetch("1UA7", format="pdb"))
    base_scaffold_AA_atoms = base_scaffold[filter_amino_acids(base_scaffold)]          # AA only
    base_scaffold_chainA_AA_atoms = base_scaffold_AA_atoms[base_scaffold_AA_atoms.chain_id == "A"] # chain A only
    
    # remove hydrogen atoms
    if "element" in base_scaffold_chainA_AA_atoms.get_annotation_categories():
        base_scaffold_chainA_AA_atoms = base_scaffold_chainA_AA_atoms[base_scaffold_chainA_AA_atoms.element != "H"]
    else:
        base_scaffold_chainA_AA_atoms = base_scaffold_chainA_AA_atoms[~np.char.startswith(base_scaffold_chainA_AA_atoms.atom_name.astype(str), "H")]
    
    start_res = base_scaffold_chainA_AA_atoms.res_id.min()
    end_res = base_scaffold_chainA_AA_atoms.res_id.max()
    base_scaffold_template = get_atomarray_in_residue_range(atoms=base_scaffold_chainA_AA_atoms, start=start_res, end=end_res, chain="A")
    



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
            residues=activesite1, 
            backbone_only=False,  # Make the inputs Ca-only
            distogram_separation=True,  # use distogram separation to calculate
            weight=5,
            name='TME_asite1'
        ),


        TemplateMatchEnergy(
            oracle=esmfold,
            template_atoms=template2,
            residues=activesite2, 
            backbone_only=False,  # Make the inputs Ca-only
            distogram_separation=True,  # use distogram separation to calculate
            weight=5,
            name='TME_asite2'
        ),


        TemplateMatchEnergy(
            oracle=esmfold,
            template_atoms=base_scaffold_template,
            residues=base_residues, 
            backbone_only=True,  # Make the inputs Ca-only
            distogram_separation=True,  # use distogram separation to calculate
            weight=5,
            name='TME_scaffold'
        ),


        bg.energies.SeparationEnergy(
            oracle=esmfold,
            residues=(activesite1, activesite2),
            # x centroid distance (Å)，target x0=20 at least, greater penality if x0<=20, no penality if x0> 20
            function=lambda x, x0=20.0: 0.0 if x >= x0 else (x - x0) ** 2, 
            weight=1.0,
        ),

        bg.energies.PTMEnergy(
            oracle=esmfold,
            weight=10,
        ),

        bg.energies.OverallPLDDTEnergy(
            oracle=esmfold,
            weight=1,
        ),

        # bg.energies.SecondaryStructureEnergy(
        #     oracle=esmfold,
        #     residues=base_residues,
        #     target_secondary_structure='alpha-helix',
        #     weight=1,
        # ),

        # let surface be hydrophilic (can also have another term to define the core to be hydrophobic)
        # bg.energies.HydrophobicEnergy(
        #     oracle=esmfold,
        #     mode='surface', 
        #     weight= -1,
            
        # ),

        # # Active site hydrophilic term (negative for hydrophilic surface)
        # bg.energies.SurfaceAreaEnergy(
        #     oracle=esmfold,
        #     residues=base_residues,  # 指定残基列表
        #     weight= -1,
        #     inheritable=False,  # 是否可继承（用于 GrandCanonical 模拟）
        #     name='SASA'
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
            'substitution': 0.5,
            'addition': 0.0,
            'removal': 0.0,
            'swap': 0.5,
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
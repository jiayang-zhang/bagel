import bagel as bg
from bagel.constants import aa_dict

base_sequence = 'ACDEFGHIKK'
print("Original sequence:", base_sequence)

# 全部可变
residues = [
    bg.Residue(name=aa, chain_ID='A', index=i, mutable=True)
    for i, aa in enumerate(base_sequence)
]
chain = bg.Chain(residues=residues)

# 建立 state，不加任何 energy term
state = bg.State(
    name='test_state',
    chains=[chain],
    energy_terms=[],    # <-- 空的 energy 列表
)

# 建立 system
system = bg.System(states=[state])

mutation_bias_no_cystein_no_lysine = {aa: 1.0 / (len(aa_dict) - 2) if (aa != 'C' and aa != 'K') else 0.0 for aa in aa_dict.keys()}
print("\nMutation bias (no Cys, no Lys):", mutation_bias_no_cystein_no_lysine)

removal_bias_no_lysine = {aa: (0.0 if aa == 'K' else 1.0) for aa in aa_dict.keys()}
print("\nRemoval bias (no Lys):", removal_bias_no_lysine)
# 建立 GrandCanonical mutator，只允许 substitution
mutator = bg.mutation.GrandCanonical(
    n_mutations=10,
    mutation_bias = mutation_bias_no_cystein_no_lysine,
    removal_bias=removal_bias_no_lysine,
    move_probabilities={
        'substitution': 1.0,
        'addition': 0.0,
        'removal': 0.0,
        'swap': 0.0
    },
)

print("\nRunning mutation...")
new_system, record = mutator.one_step(system)



# print the mutations
print("\nMutations:")
for m in record.mutations:
    print(f"  {m.move_type} at index {m.residue_index}: "
          f"{m.old_amino_acid} → {m.new_amino_acid}")

# print final sequence
print("\nOriginal sequence:", base_sequence)
print("New sequence:", new_system.states[0].chains[0].sequence)
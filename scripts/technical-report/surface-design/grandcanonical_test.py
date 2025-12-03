import sys
print("Python executable:", sys.executable)

import bagel as bg

base_sequence = 'ACDEFGHI'
print("Original sequence:", base_sequence)

# 全部可变
residues = [
    bg.Residue(name=aa, chain_ID='A', index=i, mutable=True)
    for i, aa in enumerate(base_sequence)
]
chain = bg.Chain(residues=residues)

# 建立 state，⚠️不加任何 energy term
state = bg.State(
    name='test_state',
    chains=[chain],
    energy_terms=[],    # <-- 空的 energy 列表
)

# 建立 system
system = bg.System(states=[state])

# 建立 GrandCanonical mutator，只允许 substitution
mutator = bg.mutation.GrandCanonical(
    n_mutations=2,
    move_probabilities={
        'substitution': 0.0,
        'addition': 0.0,
        'removal': 0.0,
        'swap': 1.0
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
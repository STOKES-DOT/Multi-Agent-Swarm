# MoleculeEditor command equivalence fix

The v2 pilot rejected chemistry-valid children after FLAME because its requested
commands omitted fields that the CLI's AddAtom/AddBond serializer fills with
defaults. Direct dictionary comparison treated this as an unauthorized edit.

`canonicalize_commands` now validates a transaction against its inspected parent
and returns the exact v1 command-default representation. It does not rebuild a
molecule, sanitize chemistry, change entity IDs, or ignore unexpected fields.

- add_atom: isotope=0, formal_charge=0, radical_electrons=0,
  chiral_tag=CHI_UNSPECIFIED, explicit_h_count=0, no_implicit=false,
  aromatic=false, atom_map=null.
- add_bond: aromatic follows bond_type, conjugated=false, stereo=STEREONONE,
  stereo_atom_ids=[], bond_direction=NONE.
- change_bond: aromatic=true is implicit only for an AROMATIC bond. Other
  omitted fields keep their patch semantics.
- replace_atom: omitted atom_map preserves the old map; explicit null clears it.
  These representations remain distinct. No replacement defaults are invented.
- Other commands retain their validated fields, order, references and fragments.

The proposal adapter normalizes before authorizing a transaction. The editor
provider normalizes before sending. Returned commands are normalized against the
same original parent before comparison, both in the workflow before FLAME and
at candidate acceptance. Actual element/charge/isotope/stereo changes, unknown
fields, invalid IDs, forward references, and reordered dependent commands are
still rejected.

Verification includes nine real CLI operations with request/response equality,
unit tests for null/omission semantics and invalid inputs, and a workflow test
showing a genuinely changed command is rejected without calling FLAME.

Read-only replay against the pilot's existing artifacts verified p3, p9, p6,
p1 and p5. Every artifact SHA-256 and candidate hash matched, and all five
previously rejected candidates passed corrected acceptance. Deliberately
changing the added element was rejected in every case. No FLAME rerun or
mutation of the pilot database was performed. This replay validates reuse of
the saved results; it does not retroactively change the pilot's episode status,
reflection records, pbest, or gbest.

The fix is developed separately from the running pilot source. A new process
must use the corrected revision; existing live Python processes do not reload it.

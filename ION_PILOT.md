# Explicit ions in a small Boltz-2 cofolding pilot

This is a descriptive experiment for [Boltz issue #711](https://github.com/jwohlwend/boltz/issues/711), using the unmodified released source at `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. It answers whether supplying native ions **can change these predictions**; it does not determine the checkpoint's actual training-set composition or establish a general improvement.

## Results: all 36 structures included

Two fixed protein constructs, with/without their native ions, three seeds (0, 1, 2), three diffusion samples per condition and seed. Lower C-alpha RMSD to the deposited reference is better.

| Construct | Without ions, mean / median RMSD (Å) | With ions, mean / median RMSD (Å) |
|---|---:|---:|
| 1ZNF zinc-finger core, 25 residues; 1 Zn | 1.621 / 1.641 | 1.578 / 1.559 |
| 1CLL calmodulin, 148 residues; 4 Ca | 16.864 / 17.645 | 9.848 / 14.607 |

Per-seed means over **all three** samples:

| Construct | Seed | Without ions (Å) | With ions (Å) | Difference (Å) |
|---|---:|---:|---:|---:|
| 1ZNF | 0 | 1.587 | 1.603 | +0.016 |
| 1ZNF | 1 | 1.651 | 1.584 | −0.067 |
| 1ZNF | 2 | 1.626 | 1.548 | −0.078 |
| 1CLL | 0 | 18.112 | 11.394 | −6.718 |
| 1CLL | 1 | 15.737 | 11.369 | −4.368 |
| 1CLL | 2 | 16.743 | 6.782 | −9.961 |

For the zinc finger, the small change has mixed signs across seeds. Calmodulin's global structure changes substantially, with lower mean global RMSD in all three seeds, but **large absolute errors and variation remain**. Global calmodulin RMSD is sensitive to inter-domain motion; this is not a claim of uniformly correct calcium-bound poses. With ions present, the mean absolute error in the fixed experimental donor–ion distances is 0.072 Å for 1ZNF and 0.607 Å for 1CLL. These local coordination measurements and whole-protein RMSD measure different properties.

## Protocol fixed before inference

The [plan](research/ions/plan.json), [runner](research/ions/study.py), inputs and reference mappings were frozen in local research commit `a307177` before prediction. The plan stores their hashes and upstream commit. All samples are retained; there is no confidence-based or reference-based selection.

- Stock structure inference: 3 recycles, 200 diffusion steps, 3 samples, maximum one parallel sample; single-sequence `msa: empty`, no templates, no force potentials.
- 1ZNF: remove ACE/NH2 caps (the terminal X entries in the deposited 27-residue polymer); use the same natural 25-residue core in both conditions. Input residues 1–25 map to deposited label sequence positions 2–26. Use model 1 of the deposited NMR ensemble, fixed before inference.
- 1CLL: full 148-residue sequence in both conditions. Omit waters and ethanol identically. Missing experimental coordinates are excluded from alignment.
- RMSD: all resolved experimental protein C-alpha atoms, a proper rotation and translation, reference model 1. For ions, match identical metals by optimal assignment after protein alignment. Protein donor atoms (N/O/S within 3 Å of the experimental ion) are fixed from the reference.
- Same checkpoint bytes and precision in both conditions. One H200, at most two CLI processes concurrently. Three successful processes, 36 structures, 140.45 seconds elapsed; own-process sampled peaks 3,506 MiB. These are shared-device resource records, **not a speed benchmark**.

The [analysis](research/ions/analysis.json) contains every sample, per-seed values and metrics. The `research/ions/runs/` tree contains all 36 predicted CIFs and confidence JSONs. An [independent verifier](research/ions/verify.py) reads the structures through Gemmi, checks complete protein sequences and input/reference hashes, uses SciPy rotation fitting and exhaustive ion permutations, and recovers the original metrics with maximum difference below 1e-10 Å. These are research checks, not an official upstream test suite.

## Training-data question and limits

The released PDB preprocessing applies [ExcludedLigands](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/data/filter/static/ligand.py) through [the RCSB processing script](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/scripts/process/rcsb.py). In that particular ligand filter, Zn, Ca and Mg are **not** excluded; Na and Cl **are**. Other filters also apply. This does not establish which examples/counts trained the released checkpoint; the authors need to confirm that part. See [source audit](research/ions/source-audit.json).

Both PDB constructs are old and may have appeared in training. Single-sequence mode is a deliberately restricted setting, not the recommended accuracy workflow with MSAs. Equal seeds do not imply paired diffusion noise when ions change tensor shapes. Samples within seeds/proteins are correlated: no independent-sample p-values or broad generalization claim are made. The [Boltz-2 paper](https://jeremywohlwend.com/assets/boltz2.pdf) separately notes that the **affinity module** does not explicitly handle cofactors such as ions/water/multimeric partners; this pilot concerns structure cofolding only.

## Reproduction

Use the Python 3.12 environment, pinned source and model setup in [CONTEXT300.md](CONTEXT300.md#rerun-affinity-on-the-same-frozen-structures). Source CIFs are included with hashes from RCSB: [1ZNF](https://www.rcsb.org/structure/1ZNF), [1CLL](https://www.rcsb.org/structure/1CLL). Credit belongs to their depositors and the Boltz authors. Dependencies for CPU rescoring are NumPy, SciPy and Gemmi; YAML and Boltz are required for fresh inference.

```bash
.venv/bin/python scripts/reproduce_ion_pilot.py published-score
.venv/bin/python scripts/reproduce_ion_pilot.py prepare --workdir local-ion-pilot
BOLTZ_RESEARCH_GPU=0 .venv/bin/python scripts/reproduce_ion_pilot.py run --workdir local-ion-pilot
.venv/bin/python scripts/reproduce_ion_pilot.py score --workdir local-ion-pilot
```

The helper rebases machine paths and refuses to replace an existing plan/run. Historical plan command paths remain intact as provenance. Hardware/library nondeterminism may change fresh predictions; the supplied predictions define the reported observation. Research by Arman Aubakirov with Codex assistance.

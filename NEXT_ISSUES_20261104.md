# Boltz follow-up: template parsing, coordinate export and covalent links

Three issue investigations, based on upstream `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. Each PR has an independent branch. No maintainer approval or merge is implied.

| Issue | Contribution | Status |
|---|---|---|
| [#669](https://github.com/jwohlwend/boltz/issues/669) | [PR #714](https://github.com/jwohlwend/boltz/pull/714): missing polymer-sequence metadata | Ready for review; 15 focused CPU cases |
| [#696](https://github.com/jwohlwend/boltz/issues/696) | [PR #715](https://github.com/jwohlwend/boltz/pull/715): valid PDB columns and unique chain aliases | Ready for review; 22 focused cases, 68 frozen-prediction comparisons |
| [#675](https://github.com/jwohlwend/boltz/issues/675) | [PR #716](https://github.com/jwohlwend/boltz/pull/716): explicit mmCIF connections | Ready for review; 9 focused writer cases |
| [#675](https://github.com/jwohlwend/boltz/issues/675) | [Draft #717](https://github.com/jwohlwend/boltz/pull/717): intra-chain guidance routing | **Experimental: internal-geometry regression remains**; 10 focused feature/potential cases |

These are new focused checks, **not the entire upstream test suite**. Prior community work and PR #210 are acknowledged in the relevant PRs. Research and implementation by Arman Aubakirov with Codex assistance.

## #669: reproduce the author's template crash

The supplied PDB has no SEQRES. The released parser indexes an empty polymer sequence and raises `IndexError`. The patch uses the observed residue sequence **only when metadata is absent**, warns that missing residues cannot be reconstructed, and preserves supplied full sequences. Multi-model PDB subchain renaming is also applied consistently across models.

The original file now parses as B1: 360 residues / 2,903 atoms; C1: 55 / 453. All present parsed coordinates match source positions to within 3.68e-6 Å (FP32 representation). The YAML's template mapping covers E→B1 and F→C1 across all 360 and 55 positions. Synthetic tests cover PDB/mmCIF, multiple chains/models, numbering gaps, insertion codes, mixed metadata, coordinate preservation and unresolved positions. Before: **13 fail / 2 pass**; afterward **15 pass**.

The full affinity YAML encounters a separate upstream `AllChem.Descriptors` error on this installed RDKit 2026.9.1. The original protein/ligand sequences and template settings were retained, but the affinity property was disabled for the isolated YAML template test. **Full affinity inference was not validated by this fix.**

[Baseline reproduction](reports/contributions/issue669/baseline-author-reproduction.json) · [Template-workflow result](reports/contributions/issue669/fixed-author-template-workflow.json) · [Reproduction script](research/contributions/missing_polymer_sequence_audit.py). The script downloads the author's public attachments and checks their recorded hashes. Original attachments are linked rather than redistributed. The public result encodes an unbounded template threshold as the string `unbounded`.

## #696: export the same coordinates twice

The PDB writer used the full chain name in a minimum-width one-character field. A name such as `prot1` shifts coordinate columns. Simply truncating names, as proposed in the existing [PR #210](https://github.com/jwohlwend/boltz/pull/210), can collapse A1/A2 into the same ID. The proposed writer reserves valid one-character IDs first, assigns unique aliases, records renamed IDs in application-specific REMARK rows and rejects more than 62 chains. mmCIF retains original names.

All **17 saved L1000 complex predictions** were exported in four naming configurations: native A/B, long labels, same-initial labels and a reserved-ID case. All 68 comparisons reuse identical coordinates; no model was rerun. Maximum displacement after reading long-name PDB exports changes from **1,377.361 Å** to **0.000851 Å**, within PDB rounding. All native A/B PDB outputs are byte-identical to the original writer. mmCIF is unchanged.

The 22 focused checks include both Boltz atom representations, alias collisions, 62/63-chain boundaries, CONECT/TER consistency, no mutation and the real prediction callback saving the same tensor in both formats. Initially **18 of 20** writer cases fail; two callback integration cases were added afterward.

The original issue contains images but no YAML, coordinates or seed. This demonstrates a real serialization defect; it **cannot establish the exact cause of those images**. Separate diffusion runs can also produce different structures. The [wwPDB format](https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM) specifies one chain-ID column and three coordinate decimals.

[All 68 comparisons](reports/contributions/issue696/frozen-prediction-audit.json) · [17 frozen inputs and hashes](data/pdb-format/manifest.json) · [CPU audit](research/contributions/pdb_format_audit.py).

## #675: separate connectivity metadata from geometry

The structure graph contains the declared ACE CH3 ↔ CY3 SG link. Two later paths lose it: the feature builder skips same-chain bonds before the existing connection potential, and the mmCIF writer omits the explicit connection category. These concerns are split:

- **PR #716 changes annotations only.** It records exact atom/residue/chain endpoints and measured distance through Gemmi's `_struct_conn` writer, handles polymer/non-polymer IDs and legacy atom names, skips missing endpoints, deduplicates reversed links and preserves existing connection/quality categories. It does not move atoms or change inference. An annotation-only re-export of all 24 stored pilot structures confirms unchanged arrays, expected coordinate rounding and correct endpoints.
- **Draft #717 changes guidance routing.** Explicit intra-chain `COVALENT` atom pairs reach the existing `ConnectionsPotential`; ordinary residue-internal bonds and existing inter-chain routing are retained. It changes no trained weights, force constants or inference settings. Correct routing alone does not establish correct whole-molecule chemistry.

### Frozen H200 pilot, all outcomes included

Use the issue's 16-position `GYYPRVARLIFVAIYC` construct with ACE at 1, CY3 at 16 and the declared CH3–SG bond. Add `msa: empty` identically to both versions; no templates. Released structure checkpoint; bf16 trunk / FP32 diffusion; native cuEquivariance setting. Settings: 3 recycles / 200 diffusion steps / 3 samples; seeds 0, 1, 2; native `--use_potentials` flags (FK steering and physical guidance enabled, default contact guidance retained). Model loaded once per version; Torch seeds reset before feature construction and each forward. This is a direct model/feature study, **not a CLI runtime benchmark**.

| Guided samples, n=9 per version | Original | Routing change |
|---|---:|---:|
| Mean closure distance | 5.534 Å | 1.953 Å |
| Range | 4.440–6.809 Å | 1.919–1.978 Å |
| Within existing connection-potential upper bound, 2 Å | 0/9 | 9/9 |
| Internal modified-residue bond-range violations, total | 0 | **5 across 4 samples** |

The range check uses the same original RDKit bounds expanded by 12.5% for both versions. One new ACE C–CH3 distance is **2.661 Å**, versus upper bound **1.686 Å**. Closing the declared ring can deform the internal geometry under the existing force balance. **The draft is not a complete fix for the issue's modified-peptide geometry complaints.** The follow-up ALY failure has no exact input/output here and is not claimed resolved.

Three unsteered seed-0 samples per version are bit-identical. Every input tensor hash matches except the explicit connected-atom indices. The pilot includes **24 structures total** (18 guided + 6 unsteered controls), no selection by confidence/reference and no independent-sample p-value claim. FK samples share resampling history and this is only one peptide construct.

A separate full native CLI smoke run, with both branches applied, completes with **zero failed examples**, finite confidence and one correct connection record. Its closure distance is **1.971 Å**. It uses seed 0, 3 recycles / 200 steps / **one** sample. Thus **25 GPU structures** were generated in total, across the pilot and smoke run; CPU exports/tests are not additional inference runs.

[Comparison](reports/contributions/issue675/comparison.json) · [All original samples](reports/contributions/issue675/baseline/result.json) · [All changed samples](reports/contributions/issue675/solution/result.json) · [Internal-geometry exceptions](reports/contributions/issue675/internal-geometry-limit.json) · [Annotation-only audit](reports/contributions/issue675/annotation-only-audit.json) · [Native CLI verification](reports/contributions/issue675/cli-verification.json). Every pilot CIF/NPZ, input, settings and feature/source hash file is included under those run directories. Resource timings are shared-device records, not speed comparisons.

## Reproduction

Use the pinned Python 3.12 environment, pristine upstream source and public model/CCD setup in [CONTEXT300.md](CONTEXT300.md#rerun-affinity-on-the-same-frozen-structures). The individual source commits are:

- #714: `19f2d0c98c52cc5b69f2bca75cc0eb55338f397b`;
- #715: `11ce28b2058766e3f0b5620ff76b2ea821c62222`;
- #716: `6594d2dfe7042d8740e5ac6d3edf73182a643f10`;
- combined #675 pilot source (annotation + routing): `b287f56be7340b309e11d95500810aeb5a84d94f` on `codex/enforce-explicit-covalent-bonds`;
- independent guidance-only branch: `codex/route-intrachain-covalent-guidance`, linked in Draft #717.

Clone [the implementation fork](https://github.com/AubakirovArman/boltz) into `contributions/boltz`, check out the relevant commit, and select it with `PYTHONPATH=contributions/boltz/src`. The repository's editable install may continue to point at pristine `upstream`; the explicit PYTHONPATH is required to test a fork branch.

```bash
# With the #714 source selected:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=contributions/boltz/src .venv/bin/python \
  research/contributions/missing_polymer_sequence_audit.py --output local-669.json

# With the #715 source selected; all 17 saved inputs are included:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=contributions/boltz/src .venv/bin/python \
  research/contributions/pdb_format_audit.py --output local-696.json

# First select pristine upstream, then the combined #675 source, using fresh outputs:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=upstream/src .venv/bin/python \
  research/contributions/covalent_guidance_pilot.py --output local-675-baseline
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=contributions/boltz/src .venv/bin/python \
  research/contributions/covalent_guidance_pilot.py --output local-675-solution
```

Helpers preserve previous results. Hardware/library changes can alter fresh predictions; the published artifacts define the observations. CPU fixture tests are included in each respective PR. No claim is made about binding affinity accuracy, docking quality or broad modified-residue reliability.

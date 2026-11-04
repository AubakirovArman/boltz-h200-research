# Intra-chain covalent guidance: geometry regression resolved on the validation panel

[PR #717](https://github.com/jwohlwend/boltz/pull/717) now guides explicit intra-chain covalent links while protecting the existing residue geometry and neighboring peptide bonds. This follows the [initial investigation](NEXT_ISSUES_20261104.md), where closing the link introduced other bond distortions. It addresses that observed regression; it does not establish general modified-peptide prediction accuracy or fully resolve every aspect of [issue #675](https://github.com/jwohlwend/boltz/issues/675).

Implementation and experiments by Arman Aubakirov with Codex assistance. The Boltz model, weights and existing physical potentials are the work of the Boltz authors. The issue discussion's distinction between an explicit connection and atom-level geometry bounds informed this investigation.

## What changed

Three mechanisms explained the draft's failures:

1. Explicit intra-chain `COVALENT` links were originally dropped. Routing them to connection guidance exposed a force imbalance: connection guidance had weight 0.15, whereas the existing residue geometry had weight 0.01. A separate contribution of 0.14 now protects existing geometry bounds in the affected chains, without adding a resampling weight.
2. An upper-only 2 Å connection bound also allowed excessive compression. New intra-chain links with known elements use the sum of RDKit covalent radii, buffered by ±12.5%. The ACE CH3–CY3 SG interval is 1.58375–2.03625 Å. This is an approximate, element-based interval: the YAML does not specify bond order. Unknown elements warn and fall back to the legacy bound; inter-chain links retain their existing treatment.
3. CCD bounds stop at residue boundaries. The patch adds adjacent protein C–N peptide bounds in affected chains, using the RDKit amide reference `CC(=O)NC`, and skips residue-index gaps or missing atoms.

The patch also deduplicates reversed/repeated intra-chain links, rejects self-links, and gives exactly coincident distance endpoints a finite zero subgradient. Nonzero distances retain the original analytic gradient. Old features without the new optional tensors remain supported. Unaffected chains receive no extra geometry contribution; Boltz1 and runs without physical guidance do not enable the new potential.

All changes operate during native sampling, before confidence estimation and output. No checkpoint, learned network input, diffusion schedule, external relaxation, or post-output coordinate repair was introduced. The separate [PR #716](https://github.com/jwohlwend/boltz/pull/716) adds mmCIF connection annotations; its code is not included here.

## Matched original input: all nine outputs per version

The issue's 16-position sequence `GYYPRVARLIFVAIYC`, chain B, has ACE at position 1 and CY3 at position 16, with the explicit B1 CH3 ↔ B16 SG link. `cyclic: false` is retained, with `msa: empty` added identically for the single-sequence comparison. Seeds 0, 1 and 2 each produce three samples; none is selected or discarded.

| Measurement | Pristine upstream | Initial PR draft | Final implementation |
|---|---:|---:|---:|
| Mean closure distance, Å | 5.534 | 1.953 | **1.902** |
| Closure distance range, Å | 4.440–6.809 | 1.919–1.978 | **1.862–1.944** |
| Closures inside the radii interval | 0/9 | 9/9 | **9/9** |
| Modified-residue internal bond violations | 0 | 5 | **0** |
| Modified-residue angle-distance violations | 1 | 7 | **0** |
| Adjacent peptide bond violations | 8 | 11 | **0** |

Counts are individual violating bounds across all outputs, not counts of failed structures. The initial draft's worst ACE C–CH3 bond reached 2.661 Å against a buffered upper bound of 1.686 Å. The broader audit also found adjacent peptide distortions already present in upstream predictions; these are reported rather than attributed to the draft.

## Expanded final panel

| Input | Seeds | Outputs | Closure range, Å | Internal bond / angle-distance / peptide violations |
|---|---|---:|---:|---:|
| Original ACE/CY3 | 0, 1, 2, 3, 4, 5, 2026 | 21 | 1.855–1.998 | **0 / 0 / 0** |
| ACE/CYS | 3, 4, 5, 2026 | 12 | 1.886–1.995 | **0 / 0 / 0** |
| ACE/CY3 with ALY at position 8 | 3, 4, 5, 2026 | 12 | 1.828–1.996 | **0 / 0 / 0** |

All **45/45** closures satisfy the element-based interval and are also ≤2 Å. Recorded physical potential energies are zero for these outputs under the implemented checks. ACE/CYS and the ALY-containing input are synthetic extensions, not the reporter's unavailable ALY example.

Seeds 3–5 were examined during development. Seed **2026** was recorded in a [confirmation plan](reports/contributions/issue675/refinement-20261104/final-holdout-plan.json) after the runtime code was frozen: all nine additional outputs passed. Samples produced with native FK steering are correlated; these counts are coverage observations, not independent statistical evidence of universal correctness.

## Tests, controls and native CLI

- **41 focused feature/potential tests passed**, including CPU/CUDA analytic-gradient comparisons, routing, bounds, legacy inputs, missing atoms/gaps, duplicates, self-links and zero-distance behavior. These are targeted tests, **not the full upstream suite**. [Test log](reports/contributions/issue675/refinement-20261104/final-focused-tests.log) · [test receipt](reports/contributions/issue675/refinement-20261104/final-focused-tests.json) · [static-check receipt](reports/contributions/issue675/refinement-20261104/final-static-checks.json).
- Three unsteered original-input samples and nine inter-chain control samples are bit-identical to upstream. The linear ALY control has small differences: maximum unaligned atom displacement **0.002283 Å**, maximum unaligned atom RMSD **0.000488 Å** across nine samples. A separate unchanged-upstream rerun gives maximum displacement **0.002267 Å**. This documents variation of comparable size; it does not prove its cause. Pre-existing input tensor hashes match for these controls.
- The linear ALY control still has upstream geometry violations. This is **not a general fix for standalone ALY modeling**. The inter-chain control likewise tests unchanged behavior, not validated chemistry.
- A fresh **native CLI** run processed the original and synthetic ALY inputs with zero failed examples. Independent Gemmi readback matched all output atoms to the input features and checked finite coordinates/confidences. Closure distances were **1.887 Å** and **1.905 Å**, with no tested internal bond, angle-distance or peptide-bound violations. [Readback results](reports/contributions/issue675/refinement-20261104/native-cli-verification.json) · [CLI log](reports/contributions/issue675/refinement-20261104/native-cli-final.log) · [exit receipt](reports/contributions/issue675/refinement-20261104/native-cli-final-exit.json).

The H200 runs use the released structure checkpoint, 3 recycles, 200 diffusion steps, 3 samples per seed, native `--use_potentials` steering settings, no MSA/templates, and the existing bf16-trunk/FP32-diffusion path. The native CLI smoke run uses one sample per input. cuEquivariance falls back to its reference path on this H200 for the unavailable SM100f implementation. Timings are resource records, **not a clean performance benchmark**; some jobs overlapped.

## Reproduction and retained evidence

Runtime tested: `0d41317911426090c5ac8d663a0866d025f76d5b`. Published PR head: `1f230c844437f39b8f7dbec5eff15220aefef6f1`; the only subsequent change is documentation. [Source receipt](reports/contributions/issue675/refinement-20261104/final-source.json).

Pristine baseline: `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. Initial draft: `13111c3f89917441e32fa9e1bb0ee76dfa184123`.

Use the pinned Python/environment setup in [CONTEXT300.md](CONTEXT300.md), with the structure checkpoint at `cache/boltz2_conf.ckpt` and the CCD cache at `cache/mols`. From the research repository root:

```bash
git clone https://github.com/AubakirovArman/boltz.git contributions/boltz
git -C contributions/boltz checkout 1f230c844437f39b8f7dbec5eff15220aefef6f1
PYTHONPATH=contributions/boltz/src .venv/bin/python -m pytest -q \
  contributions/boltz/tests/data/feature/test_covalent_connection_features.py \
  contributions/boltz/tests/model/potentials/test_intrachain_geometry.py

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=contributions/boltz/src .venv/bin/python \
  research/contributions/covalent_guidance_pilot.py --output local-covalent-original

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=contributions/boltz/src .venv/bin/python \
  research/contributions/covalent_refinement_validation.py \
  --output local-covalent-panel --label final --seeds 3 4 5 \
  --cases original ace_cys ace_cy3_aly linear_aly_control interchain_control

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=contributions/boltz/src .venv/bin/python \
  research/contributions/covalent_refinement_validation.py \
  --output local-covalent-confirmation --label final-seed2026 --seeds 2026 \
  --cases original ace_cys ace_cy3_aly
```

Use fresh output directories. To compare upstream, set `PYTHONPATH=upstream/src` after checking out the pinned baseline and choose different output directories. Exact native CLI arguments are preserved in the [job receipt](reports/contributions/issue675/refinement-20261104/native-cli-final-job.json); adapt absolute paths to your checkout. The [CLI readback helper](research/contributions/verify_covalent_cli.py) independently audits the saved predictions.

Recalculate the published geometry summary without a GPU:

```bash
.venv/bin/python research/contributions/covalent_refinement_summary.py \
  --root reports/contributions/issue675/refinement-20261104 \
  --output local-covalent-summary.json
```

[Machine-readable summary](reports/contributions/issue675/refinement-20261104/summary.json) · [artifact SHA-256 manifest](reports/contributions/issue675/refinement-20261104/artifact-sha256.json) · [all refinement outputs](reports/contributions/issue675/refinement-20261104).

The refinement study retains **158 new GPU-generated structures** across all stages, including failed exploratory variants and two native CLI outputs. The initial residue-only fix exposed compression; the lower-bound fix then exposed unprotected peptide bonds. Those interrupted trials and their completed outputs remain available. The 45-output final geometry panel, 41 unit tests and 158 total generated structures are different counts and must not be conflated.

No experimental reference structure was used. Satisfying these local geometry checks does not establish the correct fold, binding pose or affinity, or guarantee all other covalent chemistries. Maintainer review and broader validation remain appropriate before upstream adoption.

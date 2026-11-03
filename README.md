[Ion/no-ion cofolding pilot: 36 structures, protocol and results](ION_PILOT.md)

[Contact-union gradient fix: PR #712 and tests](CONTACT_UNION_GRADIENT.md) · [Affinity on supplied coordinates: implementation and H200 audit](KNOWN_STRUCTURE_AFFINITY.md)

# Boltz-2 on H200: CASP16 affinity reconstruction and CLI optimization

> **MW-input clarification, 2026-11-03:** the original **0.641022** result below uses RDKit MolWt of the original CSV SMILES with the fixed official correction coefficients. The CLI instead reads `Record.affinity.mw` from preprocessing. Using that exact stored feature on the same predictions gives **0.641163** (L1000 **0.536480**, L3000 **0.655632**). Thus the earlier number is a CSV-MW postprocessing calculation, not an exact reconstruction of the CLI correction input. Both definitions are now explicitly separated. The pinned RDKit also shows an implicit-hydrogen property-cache difference after `AddHs` / `RemoveHs(sanitize=False)`; see [audit and all 140 mass values](reports/casp16-issue636/MW_INPUT_CLARIFICATION.json) and [minimal reproduction](research/casp16/audit_mw_feature.py). No inference or label fitting was used for this clarification.

## New affinity experiment: observed weighted R = 0.670872

Increasing the affinity protein crop from 200 to 300 tokens gave **weighted R 0.670872**, versus **0.641163** historically and **0.638541** in the matched seed-1 stock-crop control, on all 140 fixed pairs. The corresponding uniform target mean is **0.603791**. This changes inference configuration; uncertainty and the unresolved paper evaluation convention remain explicit.

**[Protocol, results and practical reproduction instructions](CONTEXT300.md)** · [All 140 predictions](research/casp16/experiments/context300-seed1-20261103/all-predictions.csv) · [Frozen structure/input archive](data/casp16/context_inputs.tar.gz)

The 95% interval for the weighted change versus the historical result is [-0.015363, +0.074895]. Earlier negative experiments are retained. Recalculate the published observation without a GPU:

```bash
python3 scripts/reproduce_context300.py published-score
```

Research by Arman Aubakirov, prepared with Codex assistance. Boltz and its model weights are the work of the Boltz authors. This repository publishes experimental evidence, not an official Boltz benchmark or a maintainer-approved resolution.

The original reconstruction and performance studies are retained below; the new context experiment is documented above:

| Study | Measured result | Scope |
|---|---|---|
| CASP16 affinity reconstruction | Pearson R **0.641022** after the official fixed MW correction and compound-count-weighted averaging | All 140 pairs, unmodified stock inference; one completed dataset evaluation |
| CLI performance | **91.8077 → 65.8243 s**, **1.3947×**, **28.30% less time** | Median repeated L3001 processes; own sampled GPU peak **20,484 → 20,484 MiB** |
| Five-input directory performance | **204.3601 → 175.9227 s**, **1.1616×** | One stock/optimized pair; own sampled GPU peak **29,730 → 29,730 MiB** |

The 65.82-second measurement is not the runtime for 140 pairs. The optimized wrapper was not used to produce the 0.641022 quality result. The performance changes did not reduce the observed GPU-memory peak.

## CASP16: what explains most of the reported gap?

[Issue #636](https://github.com/jwohlwend/boltz/issues/636) reports approximately 0.52 versus 0.65 in Table 14 of the [Boltz-2 paper](https://jeremywohlwend.com/assets/boltz2.pdf). Two evaluation choices have substantial effects on our independent reconstruction:

1. The issue takes a uniform mean of two target correlations. Section 5 and Appendix D.2.2 describe compound-count-weighted averaging of per-assay correlations. Whether this is exactly the convention behind the CASP16 Table 14 entry still needs maintainer confirmation. Applied to the issue's rounded values, `(0.44+0.59)/2 = 0.515`, while `(17*0.44+123*0.59)/140 = 0.571786`.
2. Appendix B.5.2 describes an MW-corrected affinity ensemble. The pinned model constructor defaults to correction enabled, but the prediction CLI passes false unless `--affinity_mw_correction` is specified. Building on [ickma2311's earlier observation](https://github.com/jwohlwend/boltz/issues/636#issuecomment-6127069488), we evaluated the fixed published-code correction on the full 140 pairs. The original issue's correction setting is unknown.

```python
corrected = 1.03525938 * affinity_pred_value - 0.59992683 * MolWt**0.3 + 2.83288489
```

These are the official code's coefficients, with no fitting on CASP16. The code uses `MolWt**0.3`; we follow that pinned implementation rather than refitting the displayed paper formula. `affinity_pred_value` is the two-head affinity ensemble, not the binary binder probability.

| Target | Pairs | Raw ensemble Pearson R | Official MW-corrected R | MW-only control R |
|---|---:|---:|---:|---:|
| L1000 | 17 | 0.473280 | 0.535089 | 0.293556 |
| L3000 | 123 | 0.600116 | 0.655663 | 0.596370 |

All four calculations use exactly the same 140 raw predictions:

| Scoring | Uniform target mean | Compound-count-weighted target mean |
|---|---:|---:|
| Raw ensemble | 0.536698 | 0.584715 |
| Official MW correction | 0.595376 | **0.641022** |

Calculations:

- Aggregation change: `0.5847146339071067 - 0.5366981339774466 = +0.0480164999296601`.
- Fixed MW correction, holding weighted aggregation constant: `0.6410222018383349 - 0.5847146339071067 = +0.0563075679312282`.
- Total change between raw/uniform and MW/weighted: **+0.1043240678608883**.
- Remaining difference from 0.65: **0.0089777981616651**, whose cause is not established.

The weighted statistic is an arithmetic mean of per-target Pearson correlations with weights 17 and 123, not a pooled Pearson correlation over all molecules. Centering within targets and affine conversion of IC50 units do not explain a change in per-target Pearson R.

### How many GPU runs produced 0.641022?

**One dataset evaluation, using three CLI process attempts**, with seed 0 at each process start:

| Process attempt | Completed work | Elapsed seconds in final resource summary | Own sampled GPU peak, MiB |
|---|---|---:|---:|
| L1000 | All 17 structures and affinities | 746.2788 | 17,488 |
| L3000 initial attempt | 10 structures, no affinities; interrupted | No final summary | No final summary |
| L3000 resumed attempt | 113 additional structures, then all 123 affinities | 8,710.6446 | 31,470 |

The L1000 and initial L3000 processes overlapped on the same GPU. These elapsed times are resource records, not clean performance comparisons or total study wall time. The resumed time excludes the interrupted attempt. The original L1000 result field named `structure: 85` counts confidence files: it represents **17 pairs × 5 samples**, not 85 pairs.

Stock resume kept the first 10 completed structures without any score-based selection. Restarting the CLI resets seed 0, so its subsequent RNG sequence differs from an uninterrupted run. The interruption's cause was not established. There was no seed search, selection of a favorable repeat, coefficient fitting, or molecule exclusion to reach 0.65. Four offline scoring variants and 2,500 bootstrap resamples are **not additional GPU inference runs**.

The 2,500 paired, target-stratified bootstrap resamples (seed 20261103) give:

- MW/weighted R 95% percentile interval: **[0.532341, 0.731465]**.
- MW effect on weighted R: **[+0.035333, +0.081053]**.

These intervals describe sampling uncertainty conditional on these predictions, not variability across independent inference seeds.

### Quality configuration and checks

- Unmodified upstream commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`; no performance wrapper.
- Full L1000/L3000 sequences, 247/846 residues; frozen ColabFold env-mode MSAs; stock `max_msa_seqs=8192` retained.
- Current CASP16 affinity CSV SMILES and IC50 labels, one full protein and one ligand per input. L3000 includes all 93 PA and 30 A-only compounds.
- Structure: 10 recycles, 200 diffusion steps, 5 samples. Affinity: stock 5 recycles, 200 steps, 5 samples.
- H200 physical GPU 2; stock cuEquivariance structure kernels enabled, affinity kernels disabled; no precision changes.
- Complete ID join for all 140 pairs; 140 affinity and 700 confidence JSON files checked for finite numbers. Independent NumPy versus SciPy correlations agree within 1e-12. Source correction constants verified; FP32 versus FP64 correction changes R by less than 1e-6. Both model-checkpoint hashes unchanged after inference.

These are research consistency checks, not 840 official tests. Only Pearson R is investigated here; we do not claim reproduction of every Table 14 metric.

Input audit found 19 CSV-versus-challenge-TSV ligand differences: 17 stereochemical and two graph differences (L3027, L3101). We consistently used current CSV inputs; their effect on the paper discrepancy was not isolated. The obsolete L1000 file also contains a second assay; the primary analysis uses the current `CHYMASE_IC50_COLI_NH` assay. Exact paper/issue YAMLs, MSAs and full original predictions were unavailable. Consequently, **0.65 has not been exactly reproduced and the issue cannot be declared resolved**.

## Performance: what was accelerated?

The benchmark uses a full 846-residue L3000 protein with L3001 ligand, structure **3 recycles / 200 steps / 1 sample**, and affinity **5 / 200 / 5**. This differs from the quality study's structure settings. Checkpoints, full MSA, scientific work, precision and seed are unchanged between stock B1 and optimized runs.

1. **O1: avoid discarded truncated-normal inverse-CDF work during strict checkpoint loading.** Preserve the exact SciPy uniform random draws and ordinary Torch initialization, omit only temporary values immediately overwritten by a complete checkpoint, validate parameter coverage, restore the initializer afterward. A diagnostic attributed 36.26 of 51.92 seconds of two-model loading to this discarded work. This optimization is scoped to the pinned SciPy 1.13.1 implementation and strict prediction checkpoint loads.
2. **O2: convert constant conditioning tensors to FP32 once per diffusion sampling call**, reusing them across 200 steps. The original dictionary remains unchanged and there is no cross-input prediction cache. One L3001 conditioning tensor occupies 1,176,000,000 bytes in FP32. Precision is preserved; the observed memory peak is unchanged.
3. **O3: load the same local checkpoints with `torch.load(mmap=True)`**, preserving `map_location`, `weights_only`, weight values and dtype. This changes loading, not model math.

| Variant | Complete L3001 timing runs | Median seconds | Own sampled peak, MiB |
|---|---:|---:|---:|
| B0, stock without structure acceleration kernels | 1 | 126.2814 | 20,704 |
| B1, stock with its native structure kernels | 5 total; 4 repeats used in main median | 91.8077 for repeats | 20,484 |
| O1 | 3 | 72.9500 | 20,484 |
| O1 + O2 | 1 | 70.4598 | 20,484 |
| O1 + O3 | 1 | 67.9359 | 20,484 |
| O1 + O2 + O3 | 3 | 65.8243 | 20,484 |

Main B1 repeat times: **106.4364, 91.1366, 89.0492, 92.4789 s**. Initial B1: **112.2596 s**, recorded separately. Including all five B1 runs gives median **92.4789 s**. Combined optimized times: **67.6959, 65.5133, 65.8243 s**. Thus the headline compares repeat medians rather than the slowest initial baseline or the fastest optimized run.

`speedup = 91.80774614494294 / 65.82433038763702 = 1.3947387782646714`.

`time reduction = 100 * (1 - 65.82433038763702 / 91.80774614494294) = 28.30198632290154%`.

O2/O3 have only single-run pilot comparisons; their separate statistical contributions are not established. The combined variant has three full L3001 runs. Five inputs were also tested in the stock directory mode, one CLI process per directory: L3001, L3003, L3009, L3021 and the official 384-residue affinity example. Stock versus combined: **204.3601 → 175.9227 s**, memory **29,730 → 29,730 MiB**, one paired comparison. Loading is amortized in a directory, so the single-input gain is not a per-ligand throughput claim.

### Performance run count and correctness

The CSV contains **18 completed CLI processes**: **14 single-L3001 timing runs**, **2 five-input directory runs**, **1 diagnostic profile**, and **1 initial smoke**. The profile and smoke are excluded from performance conclusions. The smoke had an upstream example MSA mismatch and is not evidence for scientific accuracy. All individual times, including B0 and initial B1, remain in [benchmark_results.csv](reports/benchmark_results.csv). Auxiliary CPU RNG/initialization checks are not GPU inference repeats.

Optimized versus B1 outputs on the fixed five-input set have 105 numeric JSON fields and 205 array/metadata fields checked, with numeric values/array bytes equal. Exact NumPy and CPU Torch RNG states after both model loads were also compared. This is limited validation on five fixed inputs; the optimized variant was not validated on all 140 CASP16 pairs.

### Timing and GPU-memory measurement boundaries

E2E starts with a new CLI process and includes imports, model loading, input preprocessing, compilation that occurs during the call, all structure/affinity computation, and output saving. Checkpoint downloads and MSA acquisition are prepared before timing. Ordinary persistent kernel caches are retained; these are not guaranteed cold-compilation results. Each timing run writes a fresh output directory and does not reuse predictions.

VRAM is the own-process sampled NVML peak, **not total device usage or an allocator exact peak**. Sampling is approximately **100 ms for performance**, **500 ms for the quality study**; shorter peaks can be missed. MiB = 2^20 bytes. Another process occupied 6,768 MiB on the H200 and was excluded from our process-memory totals; compute contention on the shared server was not rigorously excluded. No claim of a dedicated GPU is made. CPU numerical thread settings were 8, with 2 dataloader workers.

VRAM gains are **0 MiB** in the measured B1-to-optimized comparisons. Quality-study peaks of 17,488/31,470 MiB use different sampling settings/workloads and should not be compared as an optimization result.

## Evidence and recalculation

- [All 140 predictions, labels, molecular weights and both ensemble heads](reports/casp16-issue636/predictions-and-labels.csv)
- [Full quality metrics and bootstrap intervals](reports/casp16-issue636/scoring-summary.json)
- [Combined process-attempt ledger, including the interrupted attempt](reports/run-ledger.csv)
- [All original speed measurements](reports/benchmark_results.csv), [speed summary](reports/final-summary.json)
- [Independent quality verification](reports/casp16-issue636/independent-verification.json) and [checksums](reports/casp16-issue636/output-checksums.json)
- [Russian CASP16 report](reports/casp16-issue636/REPORT_RU.md), [Russian performance report](reports/OPTIMIZATION_REPORT.md)
- Research scripts, original configs/results, frozen inputs/MSAs, and all **840 raw quality JSON outputs** are included. The new context-input archive adds the 140 original pre_affinity NPZ structures and processed inputs. Other large CIF/NPZ outputs, weights, the original paper PDF and large profiler trace are omitted; output hashes and source/download information remain available. See [publication manifest](PUBLICATION_MANIFEST.json).

Recalculate the four headline quality metrics and performance ratio with Python's standard library, without a GPU or dependencies:

```bash
python3 scripts/recalculate_published.py
```

The original `research/casp16/score.py` can additionally reproduce bootstrap and diagnostic metrics using NumPy/SciPy and the included raw affinity outputs. GPU rerun scripts preserve the research server's physical GPU UUID and original configs preserve historical absolute paths: adapt GPU selection and YAML MSA paths to your machine before rerunning inference. See [environment](manifests/environment.json), [locked requirements](configs/requirements-locked.txt) and [weight checksums](manifests/model_checksums.json). A new continuous quality run will not reproduce the historical resume's RNG sequence exactly.

Boltz source pin: `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`.
Weight repository/revision: `boltz-community/boltz-2@6fdef46d763fee7fbb83ca5501ccceff43b85607`.

Primary references: [paper](https://jeremywohlwend.com/assets/boltz2.pdf), [CASP16 official data](https://predictioncenter.org/download_area/CASP16/targets/pharma_ligands/), [pinned correction code](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/model/models/boltz2.py#L659), [pinned CLI flag](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/main.py#L992).

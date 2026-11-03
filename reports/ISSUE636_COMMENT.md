I completed an independent reconstruction of **all 140 CASP16 affinity pairs** on an H200. On the same raw predictions, applying the fixed official MW correction and compound-count-weighted averaging gives **Pearson R = 0.641022**, versus **0.536698** with the raw/uniform calculation. This explains much of the gap, but does **not** exactly reproduce 0.65.

Full results, every prediction/label, run counts, GPU-memory measurements, code, frozen inputs/MSAs and raw JSON outputs are public here: **[Boltz-2 H200 research](https://github.com/AubakirovArman/boltz-h200-research)**.

### Evaluation choices

Building on [ickma2311's MW-correction observation](https://github.com/jwohlwend/boltz/issues/636#issuecomment-6127069488), I evaluated both choices on the full dataset:

- The issue uses an equal average of two target correlations. Section 5 and Appendix D.2.2 describe compound-count-weighted per-assay correlations. For the issue's rounded numbers, the uniform mean is 0.515, whereas `(17*0.44 + 123*0.59)/140 = 0.571786`. Maintainer confirmation is still needed for the exact CASP16 Table 14 convention.
- The CLI leaves MW correction off unless `--affinity_mw_correction` is passed. I applied the pinned code's fixed formula to `affinity_pred_value`, the two-head ensemble, without fitting any coefficients on CASP16:

```python
corrected = 1.03525938 * affinity_pred_value - 0.59992683 * MolWt**0.3 + 2.83288489
```

| Same 140 predictions | Uniform target mean | Compound-count-weighted target mean |
|---|---:|---:|
| Raw ensemble | 0.536698 | 0.584715 |
| Official MW correction | 0.595376 | **0.641022** |

| Target | Pairs | Raw R | MW-corrected R |
|---|---:|---:|---:|
| L1000 | 17 | 0.473280 | 0.535089 |
| L3000 | 123 | 0.600116 | 0.655663 |

Aggregation contributes +0.048016; MW correction with weighted aggregation held constant contributes +0.056308. The remaining difference from 0.65 is **0.008978** and its cause is not established. A paired, target-stratified bootstrap with 2,500 resamples gives a 95% interval **[0.532341, 0.731465]** for the corrected weighted result and **[+0.035333, +0.081053]** for the MW effect. Resampling is conditional on these predictions, not an inference-seed repeat.

### Run count, settings and VRAM

This was **one full dataset evaluation using three CLI process attempts**, not a search over seeds or repeats:

| CLI attempt | Completed work | Recorded elapsed time | Own sampled GPU peak |
|---|---|---:|---:|
| L1000 | All 17 structures and affinities | 746.28 s | 17,488 MiB |
| L3000 initial | 10 structures, zero affinities; interrupted | No final summary | No final summary |
| L3000 resume | 113 new structures, then all 123 affinities | 8,710.64 s | 31,470 MiB |

The first two processes overlapped on the same GPU, so these are resource records, not clean speed comparisons. Resume retained the first 10 structures without score-based selection, but restarted seed 0; the RNG sequence differs from a continuous run. The interruption's cause was not established. Four scoring variants and the bootstrap required **no additional GPU inference**.

Quality inference used unmodified upstream `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`, full 247/846-residue sequences, frozen ColabFold MSAs, current CASP16 affinity CSV SMILES/IC50 labels, structure **10 recycles / 200 steps / 5 samples**, affinity **5 / 200 / 5**, seed 0 per process, stock structure kernels enabled and affinity kernels disabled. All 140 IDs are present. Independent scoring verified correlations within 1e-12; all 140 affinity plus 700 confidence JSON outputs contain finite numbers, and checkpoint hashes are unchanged. These are consistency checks, not an official upstream test suite.

Exact paper/issue inputs and full predictions are unavailable. I also found 19 CSV-versus-challenge-TSV ligand differences (17 stereochemical, two graph differences); their impact has not been isolated. No molecules, ensemble heads, coefficients or seeds were selected to reach 0.65.

### Separate CLI acceleration experiment

I also published a separate H200 performance study. This uses structure **3 / 200 / 1**, affinity **5 / 200 / 5**, so its timings must not be attached to the 140-pair quality study above.

| Workload | Stock B1 | Optimized | Speedup | Own sampled VRAM |
|---|---:|---:|---:|---:|
| L3001, repeat medians | 91.8077 s | 65.8243 s | **1.3947×; 28.30% less time** | 20,484 → 20,484 MiB |
| Five-input directory, one paired test | 204.3601 s | 175.9227 s | 1.1616× | 29,730 → 29,730 MiB |

The combination avoids discarded truncated-normal inverse-CDF initialization during strict checkpoint loading while preserving exact RNG draws, hoists unchanged FP32 conditioning conversions out of the diffusion loop, and uses memory-mapped loading of the same checkpoints. No scientific workload or precision reduction. Five-input outputs match B1 in the checked numeric JSON/array values; this optimized path was **not** used for the full 140-pair result.

The performance ledger contains **14 single-input timing runs, 2 directory runs, 1 excluded diagnostic profile and 1 excluded smoke**. The main baseline median uses four repeats (106.44, 91.14, 89.05, 92.48 s); the initial 112.26 s baseline is disclosed separately. Three combined repeats are 67.70, 65.51 and 65.82 s. Individual run data and ablations are included in the repository.

E2E includes new-process startup, imports, model loading, preprocessing, in-call compilation, computation and saved outputs; downloads/MSA acquisition precede timing. Ordinary persistent kernel caches are retained. VRAM is own-process NVML sampling (100 ms performance, 500 ms quality), which can miss short peaks. Another process's 6,768 MiB is excluded; compute contention on the shared server was not rigorously excluded. **No observed VRAM reduction is claimed** for these speed optimizations.

Could the maintainers confirm whether Table 14 used the MW correction and compound-count-weighted averaging, and share the exact CASP16 inputs/predictions to test the residual difference?

Sources: [paper](https://jeremywohlwend.com/assets/boltz2.pdf), [pinned correction code](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/model/models/boltz2.py#L659), [CLI flag](https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/src/boltz/main.py#L992), [official CASP16 data](https://predictioncenter.org/download_area/CASP16/targets/pharma_ligands/).

*Research by Arman Aubakirov, prepared with Codex assistance.*

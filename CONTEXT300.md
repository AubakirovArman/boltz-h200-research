# CASP16 affinity with a larger protein crop: observed weighted R = 0.670872

This follow-up changes the **affinity input crop**, keeping the trained Boltz-2 weights and the original predicted structures fixed. It is an exploratory configuration experiment on the same 140 CASP16 pairs. Maintainer confirmation of the exact Table 14 evaluation convention is still pending.

The headline is **compound-count-weighted per-target Pearson R**, not the uniform target mean. Both are shown below. All corrections here use the exact saved CLI `Record.affinity.mw`, with the official fixed coefficients; no coefficients or ensemble weights were fitted to CASP16 labels.

| Configuration | L1000, n=17 | L3000, n=123 | Weighted R | Uniform target mean |
|---|---:|---:|---:|---:|
| Historical original, stock crop | 0.536480 | 0.655632 | 0.641163 | 0.596056 |
| Matched seed-1 control, stock crop | 0.506387 | 0.656806 | 0.638541 | 0.581596 |
| **Larger crop, seed 1** | **0.515194** | **0.692388** | **0.670872** | **0.603791** |
| Fixed 50/50 historical-original / larger-crop ensemble | 0.536377 | 0.684580 | 0.666584 | 0.610478 |

The gain is primarily on L3000. L1000 is below the historical result. With the earlier CSV-SMILES molecular-weight definition the larger-crop weighted R is **0.670327**; definitions are kept separate in the raw tables.

## Exact change and protocol

The pinned stock affinity dataset constructs `AffinityCropper(max_tokens_protein=200)` and calls it with `max_tokens=256, max_atoms=2048`. The [small launcher](research/casp16/expanded_context_cli.py) replaces only the affinity dataset's cropper with:

- protein tokens: **200 → 300**;
- all complex tokens: **256 → 384**;
- atoms: **2048 → 3072**;
- the same stock spatial/contiguous selection algorithm, neighborhood size 10.

All original crop tokens and all ligand tokens are retained in every case. The resulting protein context is all 247 residues for L1000 and 290–300 residues for L3000. Larger token/atom caps accommodate the protein increase; the audit shows actual totals below those caps.

The full original protein sequences, atom coordinates, processed chemistry, checkpoint bytes, dtype, precision and affinity settings remain the same. The new stage uses **5 recycles / 200 diffusion steps / 5 samples**, no performance wrappers, seed **1** for each of four fixed CLI jobs: L1000 (17) and three L3000 shards (41 each). Input order matches the seed-1 stock-crop control. Changing tensor shapes can change random-number consumption despite the same seed/order.

The chosen crop and two 50/50 ensemble rules were frozen in local research commit `4f737f8c51114190f77ce5b2881aaa7ebf7ad886` before inference (see the complete original plan/source provenance in the artifacts; the full commit is recorded in publication metadata). The reported study completed **140 fresh affinity calculations**, four successful CLI processes, at most two concurrently on one H200. Existing rank-0 structures were reused; structure generation was not rerun.

Wall time was **2,148.98 s (35.82 min)** for this conditional stage. Own-process sampled NVML peaks were **17,392 / 28,412 / 21,412 / 31,508 MiB**, sampled every 0.5 s. These timings are resource records under a shared GPU workload, not an E2E speed comparison with the historical study.

## Uncertainty and prior alternatives

The observed change from the historical weighted R is **+0.029708**. The paired target-stratified bootstrap 95% interval is **[-0.015363, +0.074895]** (2,500 resamples, seed 20261103), which includes zero. Against the matched seed-1 control the interval is **[-0.009668, +0.076140]**.

The preregistered historical/new equal ensemble has weighted R **0.666584**, with change interval **[+0.002793, +0.048577]** against the historical prediction. These are unadjusted exploratory intervals after multiple experiments on just two proteins. Independent seeds and proteins are still needed; an R of 0.8 was not reached. No molecule exclusion, label-based pose selection, checkpoint fine-tuning, fitted MW coefficients, or tuned ensemble weights were used for this result.

Earlier negative results are retained: conditional seed-1 repeat, equal seed averaging, confidence-rank-1 poses, and selection by maximal `ligand_iptm`. See [all compared methods](research/casp16/experiments/AFFINITY_EXPERIMENTS_SUMMARY.json), [Russian summary](research/casp16/experiments/AFFINITY_EXPERIMENTS_SUMMARY_RU.md), and each original experiment directory. Selecting the largest observed R is not an independent estimate of generalization.

## Recalculate published numbers without a GPU

From the repository root, Python 3.12's standard library is sufficient:

```bash
python3 scripts/reproduce_context300.py published-score
```

This reads the 140 published raw affinity JSON files, checks unique/complete IDs, finite values and the two-head average, computes per-target Pearson against the published IC50 labels, and verifies both aggregation and MW definitions against the saved summary to 1e-12. It is a research consistency check, not an official upstream test suite.

## Rerun affinity on the same frozen structures

A **27,594,444-byte** [input archive](data/casp16/context_inputs.tar.gz) contains all 140 original `pre_affinity` structures and the processed input files. The [manifest](data/casp16/context_inputs_manifest.json) records every SHA256, the archive hash and the exact four case orders. Identical processed MSA contents are deduplicated as internal tar hardlinks. Model weights and canonical CCD assets are downloaded separately from the pinned public model revision. Extracted inputs use roughly 235 MB; four independent working copies require additional disk space.

The portable helper adapts machine paths and preserves byte-identical model inputs. A separate ordering adapter restores the recorded manifest order after stock preprocessing, because filesystem glob order differs between machines/disks; it makes no model or RNG changes. CPU validation against the original run is recorded in [reproduction-check.json](data/casp16/reproduction-check.json). Published `.npz`/molecule inputs are generated from public CASP16 inputs; the original large CIF/PAE/PDE outputs remain excluded.

Example installation using a compatible CUDA driver and Python **3.12**:

```bash
git clone https://github.com/AubakirovArman/boltz-h200-research.git
cd boltz-h200-research
git clone https://github.com/jwohlwend/boltz.git upstream
git -C upstream checkout b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -c configs/context-reproduction-constraints.txt -e ./upstream
mkdir -p cache
.venv/bin/python scripts/prepare_assets.py models
```

The reported full environment is in [requirements-locked.txt](configs/requirements-locked.txt) and [environment.json](manifests/environment.json). The conditional affinity model uses stock `use_kernels=False`; the separate structure-stage cuEquivariance speed study is not needed for this rerun.

Prepare with no GPU computation, then explicitly run on your selected GPU:

```bash
python3 scripts/reproduce_context300.py prepare --workdir local-context300
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/reproduce_context300.py run --workdir local-context300
python3 scripts/reproduce_context300.py score --workdir local-context300
```

Use a GPU with sufficient memory for the observed ~31 GiB maximum **per process**, plus headroom for concurrent processes and other work. The default reproduces our two-process schedule. For smaller available memory, `run --concurrency 1` lowers simultaneous residency; timing will differ. Hardware and numeric nondeterminism can also change predictions, so the supplied predictions and scoring script define the exact reported observation.

To run a fresh stock-crop comparison, use a separate directory:

```bash
python3 scripts/reproduce_context300.py prepare --context 200 --workdir local-context200
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/reproduce_context300.py run --workdir local-context200
```

For an independent seed repeat, set `--seed 2` during **prepare** with another fresh workdir. The helper preserves existing outputs and refuses to silently resume a partial run with a reset RNG.

The 140 saved structures avoid another full structure stage, but this remains a substantial GPU experiment. The published score and all input hashes can be checked on CPU before committing to it. No GPU rerun was performed merely to package this publication.

## Artifacts and provenance

- [Original preregistered plan](research/casp16/experiments/context300-seed1-20261103/plan.json), [crop audit](research/casp16/experiments/context300-seed1-20261103/crop-audit.json).
- [All predictions, controls, labels and both MW definitions](research/casp16/experiments/context300-seed1-20261103/all-predictions.csv), [metrics and intervals](research/casp16/experiments/context300-seed1-20261103/analysis.json).
- [Independent verification](research/casp16/experiments/context300-seed1-20261103/independent-verification.json), [final checkpoint integrity](research/casp16/experiments/context300-seed1-20261103/final-integrity.json).
- [Per-process resources](research/casp16/experiments/context300-seed1-20261103/result.json), [Russian report](research/casp16/experiments/context300-seed1-20261103/REPORT_RU.md).

CASP16 data: <https://predictioncenter.org/download_area/CASP16/targets/pharma_ligands/>. Boltz source/weights are credited to the Boltz authors. Research by Arman Aubakirov, prepared with Codex assistance.

# Affinity on supplied coordinates: implementation and H200 evidence

Related question: [Boltz issue #702](https://github.com/jwohlwend/boltz/issues/702).
Implementation: [fork commit 6141003](https://github.com/AubakirovArman/boltz/commit/6141003a1e73e59d4880fb5652f18d3fcbde3939), based directly on unmodified upstream `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. This branch is independent of the contact-gradient fix.

## What changes

The standard affinity forward generates coordinates and chooses a pose by IPTM.
The proposed opt-in `affinity_from_input_structure=True` path keeps the learned
trunk, trained affinity heads, ensemble and optional existing MW correction, but
passes the supplied coordinates directly to the heads. Diffusion conditioning,
sampling and confidence ranking are bypassed. The default path and checkpoint
format are unchanged. This is a Python API/example proposal, not a new flag of
the existing prediction CLI or an author-approved affinity accuracy method.

The [API documentation](https://github.com/AubakirovArman/boltz/blob/6141003a1e73e59d4880fb5652f18d3fcbde3939/docs/affinity_from_structure.md) specifies matching feature/atom order, one conformation, resolved finite FP32 coordinates, protein receptor and one small-molecule ligand. The mmCIF example uses the released parser, CCD chemistry, tokenizer, stock crop and featurizer; it refuses missing model atoms, rather than assigning zero coordinates. No assembly expansion, water handling, coordinate completion or relaxation is added.

## Checks and observed example

**32 focused CPU tests pass**. These exercise the actual forward with small
instrumented modules, including direct coordinate delivery, skipped sampling,
both heads, ensemble/MW arithmetic, default best-IPTM selection, and invalid
input rejection. This is not the entire upstream test suite.

The real-weight audit used public [1HSG](https://www.rcsb.org/structure/1HSG):
protein subchains A/B, MK1 ligand subchain C (author chain B), single-sequence
features, no waters, no assembly expansion. The retained crop has 243 tokens and
1,559 real atoms; padding is separate. The mmCIF source SHA256 is
`d2ba73b5c4be05c45275571bbb4dd042347927f91c6c05038c0df5d1ce23c268`.

With the released `boltz2_aff.ckpt`, H200, 5 recycles, standard CUDA bf16 trunk
autocast / FP32 affinity heads, no MW correction:

- Both real heads receive a **bit-equal tensor sharing the input storage**.
- Sampling and confidence calls in fixed mode: **zero**; guarded calls fail the audit.
- Explicit reference calls of the two trained heads reproduce the result **bit for bit**.
- With identical features, a different RNG seed gives bit-equal outputs.
- A 90-degree proper rotation plus translation changes none of the six affinity outputs in this example.
- The new default and pristine upstream forward are **bit-equal** for generated coordinates and affinity outputs with the same loaded weights/features/reset RNG. This regression smoke control uses **1 recycle / 20 diffusion steps / 2 samples**, not quality-study settings.

Raw fixed-pose output: `affinity_pred_value = -1.823854684829712`, binary binder
probability `0.9270296096801758`. These are model outputs, not measured affinity,
Pearson R, or evidence of improved prediction accuracy. Every check above is
recorded in [real-weight-audit.json](reports/contributions/issue702/real-weight-audit.json), with runtime source hashes.

The standalone example also completed and reproduced all six audit affinity
outputs exactly: [1hsg-h200.json](reports/contributions/issue702/1hsg-h200.json).
Its elapsed time including feature preparation and checkpoint/model loading is
**28.933 s**. The audit separately observed **25.159 s** checkpoint/model loading
and **2.323 s** fixed forward. Its maximum PyTorch allocation/reservation across
all audit operations was **3,132.84 / 3,482 MiB**; these are not NVML whole-process
peaks or a fixed-mode-only memory measurement. These are example resource
records, not a clean speed benchmark.

Two preliminary raw checkpoint-load attempts failed before inference because
legacy serialized diffusion arguments were not accepted by the released source.
The example now loads with the same explicit diffusion/trunk/steering defaults
as the prediction CLI. The successful standalone run and instrumented audit use
that final loader. There was no pose or affinity-label selection.

## Reproduce

Use the pinned environment/assets setup in [CONTEXT300.md](CONTEXT300.md#rerun-affinity-on-the-same-frozen-structures). Clone the implementation separately:

```bash
git clone https://github.com/AubakirovArman/boltz.git contributions/boltz
git -C contributions/boltz checkout 6141003a1e73e59d4880fb5652f18d3fcbde3939
PYTHONPATH=contributions/boltz/src .venv/bin/python -m pytest -q \
  contributions/boltz/tests/model/test_input_structure_affinity.py

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=contributions/boltz/src .venv/bin/python \
  contributions/boltz/examples/affinity_from_structure.py \
  reports/contributions/issue702/1HSG.cif \
  --protein-chain A --protein-chain B --ligand-chain C \
  --cache cache --output local-known-structure-example.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=contributions/boltz/src .venv/bin/python \
  research/contributions/input_structure_audit.py
```

The public audit runner reads the included source CIF and writes to a fresh
`local-known-structure-audit/` directory. Set `BOLTZ_AUDIT_OUT` to another fresh
directory for a repeat. The independent default-path comparison also requires
the pristine upstream checkout described in the setup. Source/checkpoint bytes,
software and GPU numerics matter; bit-equality here describes this recorded
environment and is not a universal cross-device guarantee.

One old complex is an engineering smoke test, potentially training-seen. Fixed
experimental geometry changes the model's standard inference workflow; it does
not establish scientific calibration or better accuracy. Missing waters,
cofactors, multimeric partners and preparation choices remain model limitations.
Credit: Boltz authors and 1HSG depositors; research and implementation by Arman
Aubakirov with Codex assistance.

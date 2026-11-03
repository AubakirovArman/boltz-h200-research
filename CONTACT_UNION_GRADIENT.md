# Contact-union guidance: sign correction

Proposed fix: [Boltz PR #712](https://github.com/jwohlwend/boltz/pull/712), for
[issue #708](https://github.com/jwohlwend/boltz/issues/708). Credit to the issue's
reporter for identifying the sign problem.

For normalized weights `w = softmax(-lambda * E)` and union energy
`F = sum(w * E)`, the derivative is
`dF/dE_i = w_i * (1 - lambda * (E_i - F))`. The released code used a plus sign.
The patch changes only that factor in the production potential.

The added focused tests compare the **actual potential** against autograd,
finite differences and energy descent: different union sizes, multiple groups,
batch sizes, lambda values, positive/negated contacts and empty constraints.
Before the patch, **26 fail / 25 pass**. After it, **51 pass on CPU** and **51
pass on H200**. These are new focused regression tests, not the entire upstream
suite.

In a fixed two-contact counterexample, the old negative-gradient step raises
energy from `0.2384058386` to `0.2390024364`; the corrected step lowers it to
`0.2382409275`. Maximum absolute error against autograd falls from
`0.4199743271` to `2.2351741791e-08`. This is evidence for the derivative fix,
not a measured whole-model affinity/structure accuracy improvement.

[Before](reports/contributions/issue708/baseline-numerics.json) ·
[After](reports/contributions/issue708/fixed-numerics.json) ·
[Numerical audit script](research/contributions/contact_union_audit.py) ·
[CPU test log](reports/contributions/issue708/fixed-cpu-tests.log) ·
[GPU test log](reports/contributions/issue708/fixed-h200-tests.log).

The baseline audit imported the pristine pinned upstream checkout separately
from the patched fork, avoiding a moving-source baseline. See runtime source
paths/hashes in both JSON files. The numerical audit script accepts an output
path; choose a fresh local file, and select the original or patched source
through `PYTHONPATH`. PR #712 includes the self-contained pytest cases.

Implementation/research by Arman Aubakirov with Codex assistance. Both the
default and zero-lambda controls remain covered. This fix does not address the
separate existing potential-underflow issue.

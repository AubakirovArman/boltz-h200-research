"""Real-weight H200 routing audit for the opt-in known-structure affinity API."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import torch
from boltz.model.models.boltz2 import Boltz2

ROOT = Path(__file__).resolve().parents[2]
FORK = ROOT / 'contributions/boltz'
SOURCE_CIF = ROOT / 'reports/contributions/issue702/1HSG.cif'
OUT = Path(os.environ.get('BOLTZ_AUDIT_OUT', ROOT / 'local-known-structure-audit'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def values(out):
    return {k: float(v.item()) for k, v in out.items() if k.startswith('affinity_')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'real-weight-audit.json').exists(), 'Use a fresh BOLTZ_AUDIT_OUT directory.'
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    example = load('known_structure_example', FORK / 'examples/affinity_from_structure.py')
    torch.manual_seed(0)
    feats, metadata = example.prepare_features(SOURCE_CIF, ROOT / 'cache', ['A', 'B'], 'C')
    device = torch.device('cuda')
    feats = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feats.items()}
    original_coords = feats['coords'].clone()
    start = time.monotonic()
    model = example.load_model(ROOT / 'cache', device)
    checkpoint_load_s = time.monotonic() - start
    calls = {'diffusion': 0, 'confidence': 0}

    def forbidden(name):
        def hook(*args, **kwargs):
            calls[name] += 1
            raise AssertionError('Fixed-structure path called ' + name)
        return hook

    guards = [model.structure_module.register_forward_pre_hook(forbidden('diffusion')),
              model.diffusion_conditioning.register_forward_pre_hook(forbidden('diffusion')),
              model.confidence_module.register_forward_pre_hook(forbidden('confidence'))]
    original_sample = model.structure_module.sample
    model.structure_module.sample = forbidden('diffusion')
    received = []

    def record_coords(module, args, kwargs):
        xyz = kwargs['x_pred']
        received.append({'module': type(module).__name__, 'bit_equal': bool(torch.equal(xyz, feats['coords'])),
                         'data_ptr_equal': xyz.data_ptr() == feats['coords'].data_ptr()})

    hooks = [head.register_forward_pre_hook(record_coords, with_kwargs=True)
             for head in (model.affinity_module1, model.affinity_module2)]
    try:
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            fixed = model(feats, recycling_steps=5, affinity_from_input_structure=True)
        torch.cuda.synchronize()
        fixed_s = time.monotonic() - start
        assert len(received) == 2 and all(x['bit_equal'] and x['data_ptr_equal'] for x in received)
        assert torch.equal(fixed['sample_atom_coords'], original_coords[:, 0])
        assert 'iptm' not in fixed
    finally:
        for handle in hooks: handle.remove()

    # Explicitly call the unchanged trained heads with the retained trunk and
    # original coordinates as an independent reference for the routing change.
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        s_inputs = model.input_embedder(feats, affinity=True)
    pad = feats['token_pad_mask'][0]
    rec = (feats['mol_type'][0] == 0) * pad
    lig = feats['affinity_token_mask'][0].bool() * pad
    cross = lig[:, None] * rec[None, :] + rec[:, None] * lig[None, :] + lig[:, None] * lig[None, :]
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        manual = [head(s_inputs=s_inputs.detach(), z=(fixed['z'] * cross[None, :, :, None]).detach(),
                       x_pred=original_coords, feats=feats, multiplicity=1, use_kernels=False)
                  for head in (model.affinity_module1, model.affinity_module2)]
    expected = (manual[0]['affinity_pred_value'] + manual[1]['affinity_pred_value']) / 2
    assert torch.equal(fixed['affinity_pred_value'], expected)

    # A second seed with exactly the same features cannot launch diffusion.
    torch.manual_seed(123)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        repeat = model(feats, recycling_steps=5, affinity_from_input_structure=True)
    for k in values(fixed): assert torch.equal(fixed[k], repeat[k]), k

    # Proper rotation by 90 degrees, followed by a translation; preserve all
    # chemical/reference features and test the supplied-geometry readout only.
    transformed = dict(feats)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=device)
    transformed['coords'] = feats['coords'] @ rotation + torch.tensor([4., -3., 1.], device=device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        rotated = model(transformed, recycling_steps=5, affinity_from_input_structure=True)
    rotation_error = max(abs(values(fixed)[k] - values(rotated)[k]) for k in values(fixed))
    assert rotation_error < 1e-4, rotation_error
    assert torch.equal(feats['coords'], original_coords)
    assert calls == {'diffusion': 0, 'confidence': 0}
    for handle in guards: handle.remove()
    model.structure_module.sample = original_sample

    # Compare the untouched upstream forward to the new default using the SAME
    # loaded weights/features and reset RNG. A shortened 20-step sampler is a
    # regression smoke test, not an accuracy/speed benchmark.
    upstream = load('boltz2_original_reference', ROOT / 'upstream/src/boltz/model/models/boltz2.py')
    kwargs = dict(recycling_steps=1, num_sampling_steps=20, diffusion_samples=2, max_parallel_samples=1,
                  run_confidence_sequentially=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        torch.manual_seed(91)
        baseline = upstream.Boltz2.forward(model, feats, **kwargs)
        torch.manual_seed(91)
        default = model(feats, **kwargs)
    assert torch.equal(baseline['sample_atom_coords'], default['sample_atom_coords'])
    for k in values(baseline): assert torch.equal(baseline[k], default[k]), k
    result = {
        'input': metadata, 'fixed_predictions': values(fixed), 'default_predictions': values(default),
        'fixed_coordinates_bit_equal': True, 'both_heads_received_same_input_storage': received,
        'diffusion_and_confidence_calls_in_fixed_mode': calls,
        'manual_head_reference_bit_equal': True, 'fixed_feature_repeat_across_seeds_bit_equal': True,
        'rotation_translation_max_affinity_output_difference': rotation_error,
        'new_default_equals_original_upstream_bitwise': True,
        'default_regression_settings': kwargs,
        'timing_scope': 'Resource records for one example, not a benchmark.',
        'load_checkpoint_s': checkpoint_load_s, 'fixed_forward_s': fixed_s,
        'peak_torch_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'peak_torch_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
        'gpu': torch.cuda.get_device_name(),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [FORK/'src/boltz/model/models/boltz2.py', FORK/'examples/affinity_from_structure.py']},
        'limitations': ['One old experimental complex, no affinity accuracy/generalization claim.',
                       'Single-sequence features; waters omitted; no structure relaxation or atom completion.',
                       'Regression control uses 20 diffusion steps and one recycle, not quality-study settings.'],
    }
    (OUT/'real-weight-audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__': main()

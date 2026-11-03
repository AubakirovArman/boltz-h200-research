"""Numerical evidence for the contact-union derivative, using real public APIs."""
import argparse,hashlib,inspect,json
from pathlib import Path
import torch
from boltz.model.potentials.potentials import ContactPotentital

parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
coords=torch.tensor([[[0.,0.,0.],[.5,0.,0.],[0.,3.,0.],[3.,3.,0.]]],requires_grad=True)
feats={'contact_pair_index':torch.tensor([[[0,2],[1,3]]]),'contact_union_index':torch.tensor([[0,0]]),
       'contact_negation_mask':torch.tensor([[True,True]]),'contact_thresholds':torch.tensor([[1.,1.]])}
potential=ContactPotentital();parameters={'union_lambda':1.}
before=potential.compute(coords,feats,parameters)
reference=torch.autograd.grad(before.sum(),coords)[0]
manual=potential.compute_gradient(coords.detach(),feats,parameters)
after=potential.compute(coords.detach()-.01*manual,feats,parameters)
source=Path(inspect.getfile(ContactPotentital))
result={'source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'torch':torch.__version__,'energy_before':before.item(),'energy_after_negative_gradient_step':after.item(),
        'energy_delta':(after-before).item(),'manual_vs_autograd_max_abs_error':(manual-reference).abs().max().item(),
        'gradient_dot_product':(manual*reference).sum().item(),
        'gradient_cosine':torch.nn.functional.cosine_similarity(manual.flatten(),reference.flatten(),dim=0).item(),
        'manual_gradient':manual.detach().tolist(),'autograd_gradient':reference.detach().tolist(),
        'scope':'ContactPotentital weighted union energy with one satisfied and one violated alternative. No model inference or benchmark quality claim.'}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if not k.endswith('_gradient')},indent=2))

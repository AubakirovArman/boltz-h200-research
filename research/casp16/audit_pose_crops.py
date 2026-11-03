"""CPU-only audit of the protein regions selected from the two saved poses."""

import json
from pathlib import Path

import numpy as np

from boltz.data.crop.affinity import AffinityCropper
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.types import Input,Record,StructureV2


HERE=Path(__file__).resolve().parent
BASE=HERE/'runs/issue-settings-csv-seed0'
EXPERIMENT=HERE/'experiments/pose-rank1-seed1-20261103'


def main():
    tokenizer=Boltz2Tokenizer();cropper=AffinityCropper();rows=[]
    for job in json.loads((EXPERIMENT/'plan.json').read_text())['jobs']:
        config=json.loads((EXPERIMENT/job['job']/'config.json').read_text())
        old=BASE/job['target']/'output'/('boltz_results_'+job['target'])
        for name in job['ids']:
            record=Record.load(old/'processed/records'/(name+'.json'));regions=[];counts=[]
            for path in [old/'predictions'/name/('pre_affinity_'+name+'.npz'),
                         Path(config['output'])/'predictions'/name/('pre_affinity_'+name+'.npz')]:
                data=Input(structure=StructureV2.load(path),msa={},record=record)
                crop=cropper.crop(tokenizer.tokenize(data),max_tokens=256,max_atoms=2048)
                protein=crop.tokens[crop.tokens['mol_type']==0]
                regions.append({(int(t['asym_id']),int(t['res_idx'])) for t in protein})
                counts.append({'protein_tokens':len(protein),'total_tokens':len(crop.tokens),
                               'atoms':int(crop.tokens['atom_num'].sum())})
            a,b=regions
            rows.append({'id':name,'target':job['target'],'rank0_counts':counts[0],'rank1_counts':counts[1],
                         'same_protein_region':a==b,'protein_region_jaccard':len(a&b)/len(a|b),
                         'rank0_only_residues':sorted(a-b),'rank1_only_residues':sorted(b-a)})
    assert len(rows)==140
    summary={t:{'n':len(g:=[r for r in rows if r['target']==t]),
                'identical_regions':sum(r['same_protein_region'] for r in g),
                'median_protein_jaccard':float(np.median([r['protein_region_jaccard'] for r in g])),
                'rank0_protein_token_range':[min(r['rank0_counts']['protein_tokens'] for r in g),max(r['rank0_counts']['protein_tokens'] for r in g)]}
             for t in ['L1000','L3000']}
    result={'scope':'Unlabelled CPU diagnostic; no changes to the frozen pose choices or inference.',
            'stock_limits':{'protein_tokens':200,'all_tokens':256,'atoms':2048},'by_target':summary,'pairs':rows}
    (EXPERIMENT/'crop-membership-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

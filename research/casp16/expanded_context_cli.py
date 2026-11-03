"""Experimental 300-residue affinity crop; all model computations stay stock.

Run this launcher in place of ``python -m boltz.main``. No performance patches,
model weights, chemical features or diffusion settings are changed. The class
is top-level so PyTorch DataLoader can also serialize it under spawn.
"""

from boltz.data.crop.affinity import AffinityCropper


PROTEIN_LIMIT = 300
TOKEN_LIMIT = 384
ATOM_LIMIT = 3072


class ExpandedAffinityCropper(AffinityCropper):
    def __init__(self):
        super().__init__(neighborhood_size=10, max_tokens_protein=PROTEIN_LIMIT)

    def crop(self, data, max_tokens, max_atoms=None):
        if (max_tokens, max_atoms) != (256, 2048):
            raise ValueError(f'Expected pinned stock crop call, got {max_tokens}, {max_atoms}')
        return super().crop(data, max_tokens=TOKEN_LIMIT, max_atoms=ATOM_LIMIT)


def main():
    import boltz.data.module.inferencev2 as inference
    # Only this module's affinity dataset constructs the replacement class.
    if inference.AffinityCropper is not AffinityCropper:
        raise RuntimeError('Affinity cropper was already replaced')
    inference.AffinityCropper = ExpandedAffinityCropper
    print(f'EXPERIMENTAL_AFFINITY_CONTEXT protein={PROTEIN_LIMIT} '
          f'tokens={TOKEN_LIMIT} atoms={ATOM_LIMIT}', flush=True)
    import boltz.main as boltz_main
    boltz_main.cli()


if __name__ == '__main__':
    main()

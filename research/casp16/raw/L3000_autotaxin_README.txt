The Autotaxin dataset, generously provided by F. Hoffmann-La Roche AG (Basel, CH), contains in total 219 protein-ligand(s) complexes after preparation.

Tasks:
- 93 protein-ligand(s) complexes: BOTH Pose (P) and Affinity (A) prediction;
- 96 protein-ligand(s) complexes: ONLY Pose (P) prediction;
- 30 protein-ligand(s) complexes: ONLY Affinity (A) prediction.

The stereochemistry is known and is provided for all the ligands for P and PA tasks but not for A tasks. The protonation state provided might not be optimal and would need to be determined. Some protein-ligand(s) complexes have more than one ligand present in the active site (22 protein-ligand(s) complexes in total). Molecules located nearby (< 4.5 A) the "main" ligand(s) and interacting with specific or non-specific interactions were also included in the SMILES files (.tsv).

The autotaxin protein carries a N-glycosylation on residue Asn497, and was considered as part of the structure (see pdb id 5m7m as example). It contains as well multiple ions: 2 zinc ions are present in the active site. These ions were also considered as part of the autotaxin structure. The position of the carbohydrate and ions won't be taken into account for the final scoring.

Reference:
@article{hunziker2022synthesis,
  title={Synthesis, characterization, and in vivo evaluation of a novel potent autotaxin-inhibitor},
  author={Hunziker, Daniel and Reinehr, Sabrina and Palmhof, Marina and Wagner, Natalie and Biniasch, Thomas and Stute, Gesa and Mattei, Patrizio and Schmitz, Petra and DiGiorgio, Patrick and Hert, J{\'e}r{\^o}me and others},
  journal={Frontiers in Pharmacology},
  volume={12},
  pages={699535},
  year={2022},
  publisher={Frontiers Media SA}
}

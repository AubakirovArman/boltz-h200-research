"""Combined candidate: exact-RNG loading, retained FP32 casts and local mmap I/O."""

import hoist_conditioning
import mmap_checkpoint


hoist_conditioning.install()
mmap_checkpoint.install(include_initialization=False)

import boltz.main as boltz_main

boltz_main.cli()

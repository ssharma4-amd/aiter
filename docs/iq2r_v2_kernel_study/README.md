# IQ2R-v2 kernel study tools

Scripts, microbenchmarks and raw results behind `docs/iq2r_v2_kernel_findings.md`. Section 16 of that document
describes every script and the paths it assumes.

- `*.sh`, `*.py`: build / timing / ISA / profiling drivers, kept as they ran (container paths
  `/workspace/experiments/iq2r-v2-20261103` and `/workspace/splits/<tree>`).
- `ubench/`, `regen/`: gfx950 instruction-rate and fp8 copy-formation microbenchmarks.
- `class_ablation/`: codebook class / book-sharing weight-error ablations on public GLM-5.3 experts, with results.
- `archive.tar.xz`: every timing log (`*.log`, `*.out`), rocprof CSVs (`prof/`), the s7 / s8 / a6 / y7 kernel
  snapshots used by `mkvar.py` / `mky.py`, and the pre-commit kernel patch. Extract next to the scripts.

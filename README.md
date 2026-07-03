# Martensitic Magnonic Crystal

Simulation code and data for **"Toward Self-Organized Magnonic Crystals in Ferromagnetic Shape Memory Alloys"** (submitted to *Journal of Applied Physics*).

## Contents

- `simulators/variant_chain.py` — 1D transfer-matrix LLG simulator (dispersion, transmission, disorder)
- `simulators/revision_figures_v2.py` — Figure generation (schematic + multi-realization disorder)
- `papers/` — LaTeX source, bibliography, and figures

## Requirements

```
pip install numpy scipy matplotlib
```

## Quick start

```python
from simulators.variant_chain import *

chain = chain_optimized(n_periods=25)
fp = compute_fingerprint(chain, freq_start=0.5, freq_stop=60, n_points=300, H_ext=6e4)

# Transmission spectrum
print(f"Stop-band fraction: {np.sum(fp['T_magnitude'] < 0.3) / len(fp['T_magnitude']):.1%}")
print(f"Deepest suppression: {np.min(fp['T_magnitude']):.1e} (>40 dB)")
```

## License

MIT

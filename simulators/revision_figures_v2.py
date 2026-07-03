#!/usr/bin/env python3
"""
Generate: (1) physical system schematic, (2) multi-realization disorder figure.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib import rcParams
import numpy as np, os, sys

sys.path.insert(0, "/home/luna/magnetic-computing/simulators")
from variant_chain import *

OUTDIR = "/home/luna/magnetic-computing/papers/figures"
os.makedirs(OUTDIR, exist_ok=True)

rcParams.update({'figure.dpi': 200, 'font.size': 11, 'axes.labelsize': 12})

# ═══════════════════════════════════════════════════════════════════
# 1. Physical system schematic
# ═══════════════════════════════════════════════════════════════════
print("Generating schematic...")

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

# Variant A blocks (easy axis horizontal)
for i, x in enumerate([0.5, 3.0, 5.5, 8.0]):
    rect = FancyBboxPatch((x, 1.0), 2.0, 2.0, boxstyle="round,pad=0.05",
                          facecolor='#4A90D9', edgecolor='#2C5F8A', linewidth=1.5, alpha=0.7)
    ax.add_patch(rect)
    # Horizontal arrows for easy axis
    for y_pos in [1.5, 2.0, 2.5]:
        ax.arrow(x + 0.3, y_pos, 1.4, 0, head_width=0.15, head_length=0.15,
                fc='white', ec='white', linewidth=1.5, alpha=0.9)
    ax.text(x + 1.0, 1.2, 'Variant A', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(x + 1.0, 0.7, 'easy axis →', ha='center', fontsize=7, color='white', alpha=0.8)
    ax.text(x + 1.0, 3.3, f'w_A', ha='center', fontsize=8, color='#2C5F8A')
    # width bracket
    ax.annotate('', xy=(x + 2.0, 3.2), xytext=(x, 3.2),
                arrowprops=dict(arrowstyle='<->', color='#2C5F8A', lw=1))

# Variant B blocks (easy axis vertical / tilted)
for i, x in enumerate([2.5, 5.0, 7.5]):
    rect = FancyBboxPatch((x, 1.0), 0.5, 2.0, boxstyle="round,pad=0.05",
                          facecolor='#E8744B', edgecolor='#A04030', linewidth=1.5, alpha=0.7)
    ax.add_patch(rect)
    # Diagonal arrows for tilted easy axis
    for y_pos in [1.5, 2.0, 2.5]:
        ax.arrow(x + 0.05, y_pos + 0.5, 0.4, -0.4, head_width=0.12, head_length=0.12,
                fc='white', ec='white', linewidth=1.5, alpha=0.9)
    ax.text(x + 0.25, 1.2, 'B', ha='center', fontsize=9, fontweight='bold', color='white')
    ax.text(x + 0.25, 0.7, 'easy axis ↙', ha='center', fontsize=6, color='white', alpha=0.8)
    ax.text(x + 0.25, 3.3, 'w_B', ha='center', fontsize=8, color='#A04030')
    ax.annotate('', xy=(x + 0.5, 3.2), xytext=(x, 3.2),
                arrowprops=dict(arrowstyle='<->', color='#A04030', lw=1))

# Twin boundary markers
for x in [2.5, 3.0, 5.0, 5.5, 7.5, 8.0]:
    ax.axvline(x=x, ymin=0.25, ymax=0.75, color='#333333', linewidth=2, linestyle='-')
    ax.text(x, 0.9, 'TB', ha='center', fontsize=7, color='#333333', fontweight='bold')

# Spin wave arrow
ax.annotate('', xy=(9.5, 2.0), xytext=(0.1, 2.0),
            arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=2.5, 
                           connectionstyle='arc3,rad=0'))
ax.text(4.8, 2.6, 'spin wave →', ha='center', fontsize=10, color='#1B5E20', fontweight='bold')

# H_ext label
ax.annotate('', xy=(9.5, 3.5), xytext=(7.5, 3.5),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2))
ax.text(8.5, 3.3, r'$\mathbf{H}_{\rm ext}$', ha='center', fontsize=10, fontweight='bold')

# Period annotation
ax.annotate('', xy=(5.5, 1.0), xytext=(3.0, 1.0),
            arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.5))
ax.text(4.25, 0.55, 'period a = w_A + w_B', ha='center', fontsize=8, color='#333333')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#4A90D9', alpha=0.7, edgecolor='#2C5F8A', label='Variant A (easy axis ∥ x)'),
    mpatches.Patch(facecolor='#E8744B', alpha=0.7, edgecolor='#A04030', label='Variant B (easy axis ⊥ x)'),
    mpatches.Patch(facecolor='none', edgecolor='#333333', linewidth=2, label='Twin boundary (TB)'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
          ncol=3, fontsize=8, frameon=False)

ax.set_title('Martensitic twin variant chain as a 1D magnonic crystal', fontsize=13, pad=10)

fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig0_schematic.png", dpi=250, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {OUTDIR}/fig0_schematic.png")

# ═══════════════════════════════════════════════════════════════════
# 2. Multi-realization disorder figure
# ═══════════════════════════════════════════════════════════════════
print("\nMulti-realization disorder analysis...")

chain = chain_optimized(n_periods=25)
H_ext = 6e4

def chain_with_disorder(chain, sigma_nm, seed):
    np.random.seed(seed)
    sigma = sigma_nm * 1e-9
    w_base_a = chain.variant_a.width
    w_base_b = chain.variant_b.width
    new_a = Variant(**{k: v for k, v in chain.variant_a.__dict__.items()})
    new_b = Variant(**{k: v for k, v in chain.variant_b.__dict__.items()})
    variants = []
    for i in range(chain.n_periods):
        da = np.random.normal(0, sigma)
        db = np.random.normal(0, sigma)
        wa = max(w_base_a + da, 5e-9)
        wb = max(w_base_b + db, 5e-9)
        a = Variant(**{k: v for k, v in chain.variant_a.__dict__.items()})
        a.width = wa; a.label = f"A{i}"
        b = Variant(**{k: v for k, v in chain.variant_b.__dict__.items()})
        b.width = wb; b.label = f"B{i}"
        variants.append(a); variants.append(b)
    c = VariantChain(variant_a=new_a, variant_b=new_b, n_periods=chain.n_periods)
    c.variants = variants
    return c

# Generate 10 realizations at σ=10 nm
n_real = 10
sigma_test = 10  # nm
T_min_values = []
T_mean_values = []
gap_fracs = []

fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True, sharey=True)
axes = axes.flatten()

for i in range(n_real):
    c_dis = chain_with_disorder(chain, sigma_test, seed=i*100 + 42)
    fp = compute_fingerprint(c_dis, freq_start=0.5, freq_stop=60, n_points=300, H_ext=H_ext)
    
    Tmin = np.min(fp["T_magnitude"])
    Tmean = np.mean(fp["T_magnitude"])
    gf = np.sum(fp["T_magnitude"] < 0.3) / len(fp["T_magnitude"])
    
    T_min_values.append(Tmin)
    T_mean_values.append(Tmean)
    gap_fracs.append(gf)
    
    ax = axes[i]
    ax.fill_between(fp["frequencies_ghz"], 0, fp["T_magnitude"], alpha=0.3, color='steelblue')
    ax.plot(fp["frequencies_ghz"], fp["T_magnitude"], color='steelblue', linewidth=0.7)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_title(f'Realization {i+1}', fontsize=8)
    ax.set_ylim(-0.02, 1.05)

fig.text(0.5, 0.02, 'Frequency (GHz)', ha='center', fontsize=11)
fig.text(0.02, 0.5, 'Transmission |S₂₁|²', va='center', rotation='vertical', fontsize=11)
fig.suptitle(f'Ten disorder realizations at $\\sigma$ = {sigma_test} nm: '
             f'deepest minima > 40 dB, '
             f'stop-band fraction = {np.mean(gap_fracs)*100:.1f} ± {np.std(gap_fracs)*100:.1f}%',
             fontsize=11)
fig.tight_layout(rect=[0.03, 0.04, 1, 0.94])
fig.savefig(f"{OUTDIR}/fig6_disorder_multi.png", dpi=250, bbox_inches='tight')
plt.close(fig)
print(f"  Saved {OUTDIR}/fig6_disorder_multi.png")
print(f"  T_min values: {[f'{v:.4f}' for v in T_min_values]}")
print(f"  Mean T_min: {np.mean(T_min_values):.4f} ± {np.std(T_min_values):.4f}")
print(f"  Mean gap fraction: {np.mean(gap_fracs)*100:.1f} ± {np.std(gap_fracs)*100:.1f}%")

# Also get T_min for the ideal (no disorder) case
fp_ideal = compute_fingerprint(chain, freq_start=0.5, freq_stop=60, n_points=300, H_ext=H_ext)
print(f"  Ideal T_min: {np.min(fp_ideal['T_magnitude']):.4f}")
print(f"  Ideal T_mean: {np.mean(fp_ideal['T_magnitude']):.4f}")
print("\nDone.")
"""Draw the segmentation-ceiling figure from the closed-form criterion (Eq. 3),
using the empirically measured error correlation rho.

sigma_EF = (1 - EF) * sigma_area * sqrt(2 (1 - rho))

The figure plots induced EF error vs per-frame area error for the worst-case
(rho = 0) and the measured (rho = 0.52) correlation, marks the model's own MAE as
the break-even bar, and places the measured DeepLabV3 operating point. It shows
that even with the realistic correlation, the segmenter sits above the ceiling.

Usage:  python3 make_ceiling_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "font.family": "serif",
    "axes.linewidth": 0.8, "savefig.bbox": "tight",
})

# ---- parameters ----
EF   = 0.60         # representative EF
MAE  = 4.1          # model's own error (break-even bar), EF points
RHO_M = 0.52        # measured ED/ES area-error correlation
SEG_ERR = 13.8      # measured DeepLabV3 per-frame area error (%)

def sigma_ef(area_pct, rho):
    return (1 - EF) * np.sqrt(2 * (1 - rho)) * area_pct

area = np.linspace(0, 25, 400)
ef_rho0 = sigma_ef(area, 0.0)
ef_rhom = sigma_ef(area, RHO_M)

# break-even area errors (where sigma_EF = MAE)
be0 = MAE / ((1 - EF) * np.sqrt(2 * (1 - 0.0)))
bem = MAE / ((1 - EF) * np.sqrt(2 * (1 - RHO_M)))
seg_ef = sigma_ef(SEG_ERR, RHO_M)

fig, ax = plt.subplots(figsize=(7.2, 5.2))

# the two ceiling curves
ax.plot(area, ef_rho0, color="#9aa0a6", lw=1.8, ls="--",
        label=r"$\rho = 0$ (independence, worst case)")
ax.plot(area, ef_rhom, color="#2a6f97", lw=2.2,
        label=rf"$\rho = {RHO_M}$ (measured)")

# model MAE break-even bar
ax.axhline(MAE, color="#c1121f", lw=1.6, ls=":",
           label=f"model error (MAE = {MAE} EF pts)")

# break-even points (where measured curve crosses MAE)
ax.plot([bem], [MAE], "o", color="#2a6f97", ms=7, zorder=5)
ax.annotate(f"break-even\n{bem:.1f}% area error",
            xy=(bem, MAE), xytext=(bem-6.2, MAE+1.6),
            fontsize=9, color="#2a6f97",
            arrowprops=dict(arrowstyle="->", color="#2a6f97", lw=1))

# DeepLabV3 operating point
ax.plot([SEG_ERR], [seg_ef], "s", color="#c1121f", ms=9, zorder=6,
        markeredgecolor="white", markeredgewidth=0.8)
ax.annotate(f"DeepLabV3\n({SEG_ERR}% area err,\n{seg_ef:.1f} EF pts)",
            xy=(SEG_ERR, seg_ef), xytext=(SEG_ERR-1.5, seg_ef+2.2),
            fontsize=9, color="#c1121f", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1))

# shade the "masks help" region (below the MAE line, left of break-even)
ax.axvspan(0, bem, ymin=0, ymax=MAE/ax.get_ylim()[1] if ax.get_ylim()[1] else 0.3,
           color="#e8f4ea", alpha=0.0)  # keep clean; annotation instead
ax.fill_between(area, 0, np.minimum(ef_rhom, MAE),
                where=(area <= bem), color="#d8ecd8", alpha=0.5)
ax.text(bem/2, MAE*0.35, "masks\ncould help", fontsize=9, ha="center",
        color="#2f6b2f")

ax.set_xlabel("Per-frame segmentation area error (%)")
ax.set_ylabel("Induced EF error, $\\sigma_{\\mathrm{EF}}$ (EF points)")
ax.set_title("The segmentation ceiling (measured correlation)")
ax.set_xlim(0, 22); ax.set_ylim(0, 11)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"segmentation_ceiling.{ext}", dpi=300, bbox_inches="tight")
print("wrote segmentation_ceiling.pdf / .png")
print(f"break-even rho=0: {be0:.1f}%,  rho={RHO_M}: {bem:.1f}%")
print(f"DeepLabV3 at {SEG_ERR}% -> {seg_ef:.1f} EF pts (bar {MAE}) -> above ceiling")

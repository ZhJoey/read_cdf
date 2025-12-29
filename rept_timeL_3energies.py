#!/usr/bin/env python3
"""
RBSP/REPT : time × L plots at specific energies (3×1 figure)

- Reads flux (FEDU/FPDU), time, L, and energy bins from a CDF file
- Collapses pitch (median by default; change to mean if needed)
- Picks the nearest energy channel to each requested energy (MeV)
- Plots three time×L panels (one per energy), same color scale (log10 flux)
"""

# ---------------------- USER SETTINGS ----------------------
FILE_PATH = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle/2017/rbspa_rel03_ect-rept-sci-L3_20170528_v5.3.0.cdf"

# Flux variable: "FEDU" (electrons) or "FPDU" (protons); set None to auto-pick.
FLUX_VAR = None

# Pitch combining: "median" (robust) or "mean" (true average)
PITCH_COMBINE = "median"

# Energies (MeV) you want to plot (nearest channel will be used)
ENERGIES_MEV = [1.8, 3.6, 4.2]

# Save the final figure?
SAVE_PLOT = True
SAVE_NAME = "rept_timeL_three_energies2.png"
# -----------------------------------------------------------

import re
import numpy as np
import matplotlib.pyplot as plt
import cdflib
from cdflib.epochs import CDFepoch

def to_dt(arr):
    return np.array(CDFepoch.to_datetime(arr))

def istp_units_to_mathtext(u: str) -> str:
    return re.sub(r'!e([+-]?\d+)!n', r'$^{\1}$', str(u))

def clean_fill(a, fill):
    b = np.array(a, dtype=float, copy=True)
    if fill is not None:
        with np.errstate(invalid="ignore"):
            b[b == fill] = np.nan
    return b

def pick_flux_var(cdf, prefer=None):
    z = list(cdf.cdf_info().zVariables)
    if prefer in z:
        return prefer
    for name in ("FEDU", "FPDU"):
        if name in z: return name
    # fallback heuristic
    cands = [v for v in z if any(t in v.lower() for t in ("flux","fedu","fpdu"))]
    cands = [v for v in cands if all(b not in v.lower() for b in ("epoch","energy","alpha","delta","labl"))]
    if not cands:
        raise RuntimeError("Could not find a flux-like variable. Set FLUX_VAR.")
    # choose highest-dim
    cands.sort(key=lambda v: np.asarray(cdf.varget(v)).ndim, reverse=True)
    return cands[0]

def classify_axis(varname: str, atts: dict) -> str:
    n = (varname or "").lower()
    units = str(atts.get("UNITS","")).lower()
    label = " ".join(str(atts.get(k,"")).lower() for k in ("FIELDNAM","LABLAXIS"))
    text = " ".join([n, units, label])
    if any(t in text for t in ("epoch","time","tt2000")): return "time"
    if any(t in text for t in ("kev","mev","gev","ev")) or "energy" in text: return "energy"
    if any(t in text for t in ("alpha","pitch","deg","degree")): return "pitch"
    return "unknown"

def combine_pitch(arr, how="median"):
    if arr.ndim == 3:
        if how == "mean": return np.nanmean(arr, axis=1)
        return np.nanmedian(arr, axis=1)
    return arr

# -------------------- OPEN & READ --------------------
print("\n[STEP] Open CDF:", FILE_PATH)
cdf = cdflib.CDF(FILE_PATH)

flux_name = pick_flux_var(cdf, FLUX_VAR)
atts_flux = cdf.varattsget(flux_name)
fillval = atts_flux.get("FILLVAL", None)
units_flux_raw = atts_flux.get("UNITS", "")
units_flux_lbl = istp_units_to_mathtext(units_flux_raw)
dep0 = atts_flux.get("DEPEND_0")
dep1 = atts_flux.get("DEPEND_1")
dep2 = atts_flux.get("DEPEND_2")
print(f"[INFO] Flux var: {flux_name} | UNITS={units_flux_raw} | FILLVAL={fillval}")
print(f"[INFO] DEPEND_0={dep0}, DEPEND_1={dep1}, DEPEND_2={dep2}")

# classify dependencies
a1 = cdf.varattsget(dep1) if dep1 else {}
a2 = cdf.varattsget(dep2) if dep2 else {}
cls1 = classify_axis(dep1, a1) if dep1 else "none"
cls2 = classify_axis(dep2, a2) if dep2 else "none"
print(f"[AXIS] DEPEND_1: {dep1} -> {cls1} (units={a1.get('UNITS','')})")
print(f"[AXIS] DEPEND_2: {dep2} -> {cls2} (units={a2.get('UNITS','')})")

energy_dep = dep1 if cls1 == "energy" else (dep2 if cls2 == "energy" else None)
pitch_dep  = dep1 if cls1 == "pitch"  else (dep2 if cls2 == "pitch"  else None)
if energy_dep is None:
    raise RuntimeError("Could not identify the energy axis from metadata.")

# axes
time = to_dt(cdf.varget(dep0))
energy_vec = np.squeeze(np.array(cdf.varget(energy_dep)))
energy_units = cdf.varattsget(energy_dep).get("UNITS","")
L = None
if "L" in cdf.cdf_info().zVariables:
    L = np.squeeze(np.array(cdf.varget("L")))
else:
    raise RuntimeError("No 'L' variable in file — required for time×L plot.")

# flux
flux_raw = cdf.varget(flux_name)  # usually (time, pitch, energy)
flux = clean_fill(flux_raw, fillval)
print("[SHAPE] flux raw:", flux.shape)
if flux.ndim == 3:
    # Ensure (time, pitch, energy) ordering
    # Decide which index is pitch & energy
    orig_pitch_axis  = 1 if pitch_dep == dep1 else (2 if pitch_dep == dep2 else None)
    orig_energy_axis = 1 if energy_dep == dep1 else (2 if energy_dep == dep2 else None)
    order = [0, orig_pitch_axis, orig_energy_axis]
    if None in order: order = [0,1,2]
    flux = np.moveaxis(flux, [0,1,2], order)
    print("[ACTION] Reordered to (time, pitch, energy). Now:", flux.shape)
elif flux.ndim == 2:
    print("[NOTE] 2D flux detected; treating as (time, energy).")
else:
    raise RuntimeError("Unexpected flux dimensionality.")

# -------------------- PITCH COMBINE --------------------
flux_te = combine_pitch(flux, how=PITCH_COMBINE)  # (time, energy)
print("[SHAPE] flux_te (time, energy):", flux_te.shape)

# -------------------- PICK ENERGY CHANNELS --------------------
energies_found = []
indices = []
for E in ENERGIES_MEV:
    idx = int(np.argmin(np.abs(energy_vec - E)))
    indices.append(idx)
    energies_found.append(float(energy_vec[idx]))
print(f"[INFO] Requested energies (MeV): {ENERGIES_MEV}")
print(f"[INFO] Using nearest channel centers (MeV): {energies_found}")
print(f"[INFO] Channel indices: {indices}")

# Build log10 flux arrays for each energy (mask non-positive)
log_flux_list = []
for idx in indices:
    f = flux_te[:, idx]  # (time,)
    f = np.asarray(f, dtype=float)
    f[~np.isfinite(f)] = np.nan
    mask_pos = f > 0
    logf = np.full_like(f, np.nan, dtype=float)
    logf[mask_pos] = np.log10(f[mask_pos])
    log_flux_list.append(logf)

# Common color scale across all three panels (ignore NaNs)
all_vals = np.concatenate([lf[np.isfinite(lf)] for lf in log_flux_list]) if any(np.isfinite(lf).any() for lf in log_flux_list) else np.array([])
if all_vals.size > 0:
    vmin = np.nanpercentile(all_vals, 5)
    vmax = np.nanpercentile(all_vals, 95)
else:
    vmin, vmax = 0, 1  # fallback

# -------------------- COLOR SCALE --------------------
vmin, vmax = 1, 4  # fixed range for log10(flux)
cmap = "jet"

print(f"[SCALE] log10 color scale vmin={vmin:.3g}, vmax={vmax:.3g}")

# -------------------- PLOT: 3×1 FIGURE --------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

for ax, logf, E in zip(axes, log_flux_list, energies_found):
    ok = np.isfinite(logf)
    sc = ax.scatter(time[ok], L[ok], c=logf[ok], s=4, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylabel("L-shell")
    ax.set_title(f"{flux_name}: L vs Time @ ~{E:.2f} MeV (pitch-{PITCH_COMBINE})")

# Make space for colorbar
fig.subplots_adjust(right=0.85, hspace=0.3)

# Add a slim colorbar
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(sc, cax=cbar_ax, label="log$_{10}$ " + (units_flux_lbl or "Flux"))

axes[-1].set_xlabel("Time (UTC)")

if SAVE_PLOT:
    fig.savefig(SAVE_NAME, dpi=150)
    print(f"[SAVED] {SAVE_NAME}")

plt.show()


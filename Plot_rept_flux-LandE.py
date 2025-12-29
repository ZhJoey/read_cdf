#!/usr/bin/env python3
"""
RBSP/REPT flux plots (easy to debug):
  - Time × Energy spectrogram (pitch-combined)
  - Time × L-shell colored by flux averaged over an energy window (pitch-combined)

Edit the USER SETTINGS block below and run.
"""

# ------------------------- USER SETTINGS -------------------------
FILE_PATH = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle/2017/rbspa_rel03_ect-rept-sci-L3_20170528_v5.3.0.cdf"

# Which flux variable? Set to "FEDU" (electrons) or "FPDU" (protons).
# If None, the script will try to pick automatically.
FLUX_VAR = None

# Pitch combining method: "mean" (true average) or "median" (robust)
PITCH_COMBINE = "median"

# What to plot:
PLOT_MODE = "time_L"   # options: "time_energy" or "time_L"

# If PLOT_MODE == "time_L": choose an energy window (MeV) to average over
ENERGY_WINDOW = (1.8, 2.5)   # e.g., average ~1.8–2.5 MeV
# or to use a single channel, set both bounds the same (e.g., (1.8, 1.8))

# Save figures?
SAVE_PLOTS = True
SAVE_PREFIX = "rept_flux"
# ----------------------------------------------------------------

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cdflib
from cdflib.epochs import CDFepoch

# ---------- helpers ----------
def to_dt(arr):
    return np.array(CDFepoch.to_datetime(arr))

def clean_fill(a, fill):
    b = np.array(a, dtype=float, copy=True)
    if fill is not None:
        with np.errstate(invalid="ignore"):
            b[b == fill] = np.nan
    return b

def istp_units_to_mathtext(u: str) -> str:
    """Convert ISTP !e...!n markup to Matplotlib mathtext superscripts."""
    return re.sub(r'!e([+-]?\d+)!n', r'$^{\1}$', str(u))

def choose_flux_var(cdf, prefer=None):
    zvars = list(cdf.cdf_info().zVariables)
    print("[INFO] zVariables:", zvars)
    if prefer and prefer in zvars:
        print(f"[INFO] Using user-selected flux var: {prefer}")
        return prefer
    for name in ("FEDU", "FPDU"):
        if name in zvars:
            print(f"[INFO] Using common flux var: {name}")
            return name
    # heuristic
    cands = [v for v in zvars if any(t in v.lower() for t in ("flux","fedu","fpdu"))]
    bad   = ("epoch","energy","alpha","delta","labl")
    cands = [v for v in cands if all(b not in v.lower() for b in bad)]
    if not cands:
        raise RuntimeError("No flux-like variable found. Set FLUX_VAR explicitly.")
    cands.sort(key=lambda v: np.asarray(cdf.varget(v)).ndim, reverse=True)
    print(f"[INFO] Heuristic picked: {cands[0]}")
    return cands[0]

def classify_axis(varname: str, atts: dict) -> str:
    """Return 'time','energy','pitch','unknown' using names/units."""
    n = (varname or "").lower()
    units = str(atts.get("UNITS", "")).lower()
    label = " ".join(str(atts.get(k, "")).lower() for k in ("FIELDNAM","LABLAXIS"))
    text = " ".join([n, units, label])

    if any(tok in text for tok in ("tt2000","epoch","time")):
        return "time"
    if any(tok in text for tok in ("kev","mev","gev","ev")) or "energy" in text:
        return "energy"
    if "alpha" in text or "pitch" in text or "deg" in text or "degree" in text:
        return "pitch"
    return "unknown"

def combine_pitch(arr, axis=1, how="median"):
    if how == "mean":
        return np.nanmean(arr, axis=axis)
    return np.nanmedian(arr, axis=axis)

# ---------- open & map ----------
print("\n=== STEP 1: Open file ===")
print("FILE_PATH:", FILE_PATH)
cdf = cdflib.CDF(FILE_PATH)
info = cdf.cdf_info()
print("[INFO] CDF info:\n", info)

print("\n=== STEP 2: Choose flux variable ===")
flux_name = choose_flux_var(cdf, prefer=FLUX_VAR)
atts_flux = cdf.varattsget(flux_name)
units_flux_raw = atts_flux.get("UNITS", "")
units_flux_lbl = istp_units_to_mathtext(units_flux_raw)
fillval = atts_flux.get("FILLVAL", None)
dep0 = atts_flux.get("DEPEND_0", None)  # expected time
dep1 = atts_flux.get("DEPEND_1", None)  # pitch or energy
dep2 = atts_flux.get("DEPEND_2", None)  # energy or pitch
print(f"[INFO] {flux_name} UNITS={units_flux_raw} FILLVAL={fillval}")
print(f"[INFO] DEPEND_0={dep0} DEPEND_1={dep1} DEPEND_2={dep2}")

# dependencies
a1 = cdf.varattsget(dep1) if dep1 else {}
a2 = cdf.varattsget(dep2) if dep2 else {}
cls1 = classify_axis(dep1, a1) if dep1 else "none"
cls2 = classify_axis(dep2, a2) if dep2 else "none"
print(f"[AXIS] DEPEND_1 {dep1} -> {cls1} (units={a1.get('UNITS','')})")
print(f"[AXIS] DEPEND_2 {dep2} -> {cls2} (units={a2.get('UNITS','')})")

# map which is energy/pitch
energy_dep = dep1 if cls1 == "energy" else (dep2 if cls2 == "energy" else None)
pitch_dep  = dep1 if cls1 == "pitch"  else (dep2 if cls2 == "pitch"  else None)

# load axis arrays
time = to_dt(cdf.varget(dep0)) if dep0 else None
energy_vec = np.squeeze(np.array(cdf.varget(energy_dep))) if energy_dep else None
pitch_vec  = np.squeeze(np.array(cdf.varget(pitch_dep)))  if pitch_dep  else None
energy_units = (cdf.varattsget(energy_dep).get("UNITS","") if energy_dep else "")
pitch_units  = (cdf.varattsget(pitch_dep).get("UNITS","")   if pitch_dep  else "")
print(f"[DECISION] energy_dep={energy_dep} units={energy_units} | pitch_dep={pitch_dep} units={pitch_units}")

# load L-shell (if available)
L = None
if "L" in cdf.cdf_info().zVariables:
    L = np.squeeze(np.array(cdf.varget("L")))
    print("[INFO] Loaded L-shell, shape:", L.shape)
else:
    print("[WARN] No 'L' variable found; time_L mode will not work.")

# load flux
flux_raw = cdf.varget(flux_name)  # expected (time, pitch, energy)
flux = clean_fill(flux_raw, fillval)
print("Flux raw shape:", np.shape(flux))

# figure out indices
orig_pitch_axis  = 1 if pitch_dep  == dep1 else (2 if pitch_dep  == dep2 else None)
orig_energy_axis = 1 if energy_dep == dep1 else (2 if energy_dep == dep2 else None)
print(f"[INDEX] orig_pitch_axis={orig_pitch_axis} orig_energy_axis={orig_energy_axis}")

# sanity reshaping: we want (time, pitch, energy) ordering for simplicity
if flux.ndim == 3:
    order = [0, orig_pitch_axis, orig_energy_axis]
    if None in order:
        # fall back: assume (time, pitch, energy)
        order = [0, 1, 2]
    flux = np.moveaxis(flux, [0,1,2], order)
    print("[ACTION] Reordered flux to (time, pitch, energy). New shape:", flux.shape)
elif flux.ndim == 2:
    # assume (time, energy) no pitch dimension
    print("[NOTE] 2D flux detected; treating as (time, energy).")
else:
    raise RuntimeError("Unexpected flux dimensionality.")

# ---------- PLOT MODE: TIME × ENERGY ----------
if PLOT_MODE == "time_energy":
    print("\n=== MODE: time × energy spectrogram (pitch-combined) ===")
    if flux.ndim == 3:
        flux2d = combine_pitch(flux, axis=1, how=PITCH_COMBINE)  # (time, energy)
    else:
        flux2d = flux  # already (time, energy)
    print("flux2d shape (time, energy):", flux2d.shape)

    # energy axis
    if energy_vec is not None and energy_vec.size == flux2d.shape[1]:
        y_energy = energy_vec
        energy_label = f"Energy ({energy_units})" if energy_units else "Energy"
    else:
        y_energy = np.arange(flux2d.shape[1])
        energy_label = "Energy (channel index)"
        print("[WARN] Using channel index for energy (vector missing/mismatch).")

    # colors: log scaling when positive values exist
    Z = flux2d.astype(float)
    finite_pos = Z[np.isfinite(Z) & (Z > 0)]
    norm = None
    if finite_pos.size > 50:
        vmin = np.nanpercentile(finite_pos, 5)
        vmax = np.nanpercentile(finite_pos, 95)
        norm = LogNorm(vmin=max(vmin, 1e-300), vmax=vmax)

    plt.figure(figsize=(10, 4.8))
    plt.pcolormesh(time, y_energy, Z.T, shading="auto", norm=norm)
    cbar = plt.colorbar()
    cbar.set_label(units_flux_lbl or "Flux")
    plt.xlabel("Time (UTC)")
    plt.ylabel(energy_label)
    plt.title(f"{flux_name}: pitch-{PITCH_COMBINE} spectrogram")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{SAVE_PREFIX}_time_energy.png", dpi=150)
        print(f"[SAVED] {SAVE_PREFIX}_time_energy.png")
    plt.show()

# ---------- PLOT MODE: TIME × L (energy-averaged) ----------
elif PLOT_MODE == "time_L":
    print("\n=== MODE: time × L colored by energy-averaged flux (pitch-combined) ===")
    if L is None:
        raise RuntimeError("No 'L' variable in file; cannot make time × L plot.")

    # 1) pitch combine first
    if flux.ndim == 3:
        flux_time_energy = combine_pitch(flux, axis=1, how=PITCH_COMBINE)  # (time, energy)
    else:
        flux_time_energy = flux  # (time, energy)
    print("flux_time_energy shape:", flux_time_energy.shape)

    # 2) select energy indices within ENERGY_WINDOW
    if energy_vec is None or energy_vec.ndim != 1:
        raise RuntimeError("Cannot select ENERGY_WINDOW: energy vector missing or not 1D.")
    E_lo, E_hi = ENERGY_WINDOW
    if E_lo > E_hi:  # swap if user reversed
        E_lo, E_hi = E_hi, E_lo
    e_mask = (energy_vec >= E_lo) & (energy_vec <= E_hi)
    e_idx = np.where(e_mask)[0]
    if e_idx.size == 0:
        # pick the nearest single channel to E_lo if window empty
        e_idx = np.array([np.argmin(np.abs(energy_vec - E_lo))])
        print(f"[NOTE] No channels in window {ENERGY_WINDOW}; using nearest channel:", energy_vec[e_idx[0]])
    print(f"[INFO] Selected {e_idx.size} energy bins in window {E_lo}–{E_hi} MeV:",
          energy_vec[e_idx])

    # 3) average over selected energies -> one flux per time
    flux_time = np.nanmean(flux_time_energy[:, e_idx], axis=1)  # (time,)
    # (If you want median over energy too, replace with np.nanmedian)

    # 4) Build color values (log10), mask nonpositive
    color = np.full_like(flux_time, np.nan, dtype=float)
    pos = flux_time > 0
    color[pos] = np.log10(flux_time[pos])

    # 5) Scatter (irregular grid in L vs time)
    plt.figure(figsize=(10, 4.8))
    sc = plt.scatter(time[pos], L[pos], c=color[pos], s=4, cmap="jet")
    cbar = plt.colorbar(sc, label="log10 " + (units_flux_lbl or "Flux"))
    plt.xlabel("Time (UTC)")
    plt.ylabel("L-shell")
    plt.title(f"{flux_name}: pitch-{PITCH_COMBINE}, E∈[{E_lo}, {E_hi}] MeV (avg)")
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig(f"{SAVE_PREFIX}_time_L.png", dpi=150)
        print(f"[SAVED] {SAVE_PREFIX}_time_L.png")
    plt.show()

else:
    raise ValueError("PLOT_MODE must be 'time_energy' or 'time_L'")

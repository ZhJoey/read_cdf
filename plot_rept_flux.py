#!/usr/bin/env python3
"""
REPT flux plotter (robust axis handling).

Fixes:
- Tracks pitch/energy axes BEFORE and AFTER collapsing pitch
- Avoids moveaxis() with invalid axis indices
- Uses true energy vector from the DEPEND classified as energy
- Converts ISTP unit markup to Matplotlib-friendly label
"""

# ------------------ USER SETTINGS ------------------
FILE_PATH = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle/2017/rbspa_rel03_ect-rept-sci-L3_20170528_v5.3.0.cdf"
FLUX_VAR   = None          # e.g., "FEDU" or "FPDU"; None = auto
PITCH_COLLAPSE = "median"  # "median" or "mean"
ENERGY_INDEX   = None      # energy-bin index to highlight; None = auto
TIME_INDEX     = None      # time index to highlight; None = auto
SAVE_PLOTS     = True
SAVE_PREFIX    = "rept_flux3"
# ---------------------------------------------------

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
    for name in ("FEDU", "FPDU"):  # common REPT vars
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

def collapse(arr, axis, how="median"):
    if axis is None or arr.ndim <= axis:
        return arr
    if how == "mean":
        return np.nanmean(arr, axis=axis)
    return np.nanmedian(arr, axis=axis)

# ---------- main ----------
def main():
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
    fillval    = atts_flux.get("FILLVAL", None)
    dep0 = atts_flux.get("DEPEND_0", None)  # expected time
    dep1 = atts_flux.get("DEPEND_1", None)
    dep2 = atts_flux.get("DEPEND_2", None)
    print(f"[INFO] {flux_name} UNITS={units_flux_raw}  FILLVAL={fillval}")
    print(f"[INFO] DEPEND_0={dep0}  DEPEND_1={dep1}  DEPEND_2={dep2}")

    # --- classify dependencies ---
    print("\n=== STEP 3: Classify axes ===")
    # DEPEND_0 is time by definition for ISTP CDFs
    cls0 = "time"
    a1 = cdf.varattsget(dep1) if dep1 else {}
    a2 = cdf.varattsget(dep2) if dep2 else {}
    cls1 = classify_axis(dep1, a1) if dep1 else "none"
    cls2 = classify_axis(dep2, a2) if dep2 else "none"
    print(f"[AXIS] DEPEND_0: {dep0} -> time")
    print(f"[AXIS] DEPEND_1: {dep1} -> {cls1} (units={a1.get('UNITS','')})")
    print(f"[AXIS] DEPEND_2: {dep2} -> {cls2} (units={a2.get('UNITS','')})")

    # map which depend is energy/pitch
    energy_dep = dep1 if cls1 == "energy" else (dep2 if cls2 == "energy" else None)
    pitch_dep  = dep1 if cls1 == "pitch"  else (dep2 if cls2 == "pitch"  else None)

    # --- load axes ---
    time_cdf = cdf.varget(dep0) if dep0 else None
    time = to_dt(time_cdf) if time_cdf is not None else None
    energy_vec = np.squeeze(np.array(cdf.varget(energy_dep))) if energy_dep else None
    pitch_vec  = np.squeeze(np.array(cdf.varget(pitch_dep)))  if pitch_dep  else None
    energy_units = (cdf.varattsget(energy_dep).get("UNITS","") if energy_dep else "")
    pitch_units  = (cdf.varattsget(pitch_dep).get("UNITS","")   if pitch_dep  else "")

    print(f"[DECISION] energy_dep={energy_dep} units={energy_units}  | pitch_dep={pitch_dep} units={pitch_units}")

    # --- load flux and clean fill ---
    flux_raw = cdf.varget(flux_name)
    flux = clean_fill(flux_raw, fillval)
    print("Flux shape (raw):", flux.shape)

    # Determine ORIGINAL axis indices for energy & pitch BEFORE collapsing:
    # flux dims are [time, DEPEND_1?, DEPEND_2?]
    orig_energy_axis = None
    orig_pitch_axis  = None
    if flux.ndim >= 2:
        if cls1 == "energy": orig_energy_axis = 1
        if cls2 == "energy": orig_energy_axis = 2
        if cls1 == "pitch":  orig_pitch_axis  = 1
        if cls2 == "pitch":  orig_pitch_axis  = 2

    print(f"[INDEX] orig_energy_axis={orig_energy_axis}  orig_pitch_axis={orig_pitch_axis}")

    # Collapse pitch if present
    if orig_pitch_axis is not None and flux.ndim >= 3:
        print(f"[ACTION] Collapsing pitch at axis={orig_pitch_axis} using {PITCH_COLLAPSE}")
        flux_collapsed = collapse(flux, axis=orig_pitch_axis, how=PITCH_COLLAPSE)
    else:
        print("[ACTION] No 3D pitch axis to collapse (using flux as-is).")
        flux_collapsed = flux

    print("Flux shape after pitch collapse:", flux_collapsed.shape)

    # Compute energy axis index AFTER collapsing
    if flux.ndim >= 3 and orig_energy_axis is not None and orig_pitch_axis is not None:
        # If we removed an axis before the energy axis, energy index shifts by -1
        if orig_pitch_axis < orig_energy_axis:
            energy_axis_after = orig_energy_axis - 1
        else:
            energy_axis_after = orig_energy_axis
    elif flux_collapsed.ndim == 2:
        # With 2D (time, something), assume axis 1 is energy
        energy_axis_after = 1
    else:
        # Fallback (rare 1D cases)
        energy_axis_after = 0

    print(f"[INDEX] energy_axis_after={energy_axis_after}")

    # Ensure final array is (time, energy). If not, move it safely.
    if flux_collapsed.ndim == 2 and energy_axis_after in (0,1):
        if energy_axis_after == 1:
            flux_2d = flux_collapsed
        else:
            # energy is axis 0 -> swap to 1
            flux_2d = np.moveaxis(flux_collapsed, 0, 1)
            print("[ACTION] Swapped axes to make shape (time, energy). New shape:", flux_2d.shape)
    elif flux_collapsed.ndim == 3:
        # Unexpected: still 3D — pick axis order to (time, energy, other)
        order = [0, energy_axis_after] + [ax for ax in (1,2) if ax not in (0, energy_axis_after)]
        flux_reordered = np.moveaxis(flux_collapsed, range(3), order)
        # collapse the remaining axis
        flux_2d = np.nanmedian(flux_reordered, axis=2)
        print("[ACTION] Reordered & collapsed to 2D. New shape:", flux_2d.shape)
    else:
        flux_2d = np.atleast_2d(flux_collapsed)
        if flux_2d.shape[0] == 1 and time is not None and len(time) == flux_2d.shape[1]:
            # transpose if needed to (time, energy)
            flux_2d = flux_2d.T

    # Build Y axis for energy
    if energy_vec is not None and energy_vec.ndim == 1 and energy_vec.size == flux_2d.shape[1]:
        y_energy = energy_vec
        energy_label = f"Energy ({energy_units})" if energy_units else "Energy"
        print("[PLOT] Using DEPEND energy vector:", energy_dep, "| len=", len(y_energy))
    else:
        y_energy = np.arange(flux_2d.shape[1])
        energy_label = "Energy (channel index)"
        print("[PLOT] Using channel index for energy (missing or size mismatch).")

    # Indices to highlight
    ei = ENERGY_INDEX if ENERGY_INDEX is not None else min(5, flux_2d.shape[1]-1)
    ti = TIME_INDEX   if TIME_INDEX   is not None else flux_2d.shape[0] // 8

    print("\n=== SUMMARY ===")
    print("Time length:", len(time) if time is not None else None)
    print("Energy length:", len(y_energy))
    print("Flux 2D shape (time, energy):", flux_2d.shape)
    print("Highlight: energy idx =", ei, "time idx =", ti)
    print("================\n")

    # ---------- Plot 1: Spectrogram ----------
    Z = flux_2d.astype(float)
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
    plt.title(f"{flux_name}: pitch-averaged spectrogram")
    plt.tight_layout()
    if SAVE_PLOTS:
        fname = f"{SAVE_PREFIX}_spectrogram.png"
        plt.savefig(fname, dpi=150)
        print(f"[SAVED] {fname}")
    plt.show()

    # ---------- Plot 2: Time series at one energy ----------
    plt.figure(figsize=(10, 3.6))
    plt.plot(time, Z[:, ei], lw=0.9)
    plt.xlabel("Time (UTC)")
    plt.ylabel(units_flux_lbl or "Flux")
    try:
        e_txt = f"{np.atleast_1d(y_energy)[ei]:.3g}"
    except Exception:
        e_txt = str(np.atleast_1d(y_energy)[ei])
    plt.title(f"{flux_name}: time series @ energy idx {ei} (~{e_txt})")
    plt.tight_layout()
    if SAVE_PLOTS:
        fname = f"{SAVE_PREFIX}_timeseries.png"
        plt.savefig(fname, dpi=150)
        print(f"[SAVED] {fname}")
    plt.show()

    # ---------- Plot 3: Spectrum at one time ----------
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(y_energy, Z[ti, :], marker="o", ms=3, lw=0.9)
    plt.xlabel(energy_label)
    plt.ylabel(units_flux_lbl or "Flux")
    plt.title(f"{flux_name}: spectrum @ time idx {ti}\n{time[ti]}")
    plt.tight_layout()
    if SAVE_PLOTS:
        fname = f"{SAVE_PREFIX}_spectrum.png"
        plt.savefig(fname, dpi=150)
        print(f"[SAVED] {fname}")
    plt.show()

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()


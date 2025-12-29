#!/usr/bin/env python3
"""
RBSP/REPT: Time × L plots over a DATE RANGE (3×1 panels at chosen energies)

- Input: start/end dates (inclusive), base directory, file pattern components
- For each day:
    * Find the CDF file (optionally choose the latest version)
    * Load FEDU/FPDU + Epoch, L, energy axis
    * Replace FILLVAL with NaN
    * Combine pitch (median or mean)
- Concatenate in time and plot 3 energies (nearest channels) in one 3×1 figure
- Slim colorbar at the right, fixed log10 range

Requires: cdflib, numpy, matplotlib
"""

# ========================== USER SETTINGS ==========================
# Date range (inclusive)
START_DATE = "2017-05-20"
END_DATE   = "2017-09-30"

# Where the files live. Often there's a year subfolder.
# Example layout:
#   BASE_DIR/2017/rbspa_rel03_ect-rept-sci-L3_20170527_v5.3.0.cdf
BASE_DIR = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle"

# File name components (edit if your naming differs)
PROBE   = "rbspa"                # "rbspa" or "rbspb"
RELEASE = "rel03"                # e.g., "rel03" or "rel04"
PRODUCT = "ect-rept-sci-L3"      # product string
CHOOSE_LATEST_VERSION = True     # if multiple v*.cdf exist for a day, pick the newest

# Flux variable: set "FEDU" (electrons) or "FPDU" (protons);
# if None, auto-pick (prefers FEDU then FPDU)
FLUX_VAR = None

# Pitch combining across bins: "median" (robust) or "mean" (true average)
PITCH_COMBINE = "median"

# Energies (MeV) to plot (nearest channel used)
ENERGIES_MEV = [1.8, 3.6, 4.2]

# Colorbar & style
LOG_VMIN, LOG_VMAX = 1, 4.0  # log10(flux) range (e.g., 1→10, 4→10^4)
CMAP = "jet"                    # "viridis", "jet", etc.

# Save the final figure?
SAVE_PLOT = True
SAVE_NAME = f"rept_timeL_{START_DATE}_{END_DATE}_3energies.png"
# ==================================================================

import re
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cdflib
from cdflib.epochs import CDFepoch
from glob import glob

# ---------- helpers ----------
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
        return np.nanmean(arr, axis=1) if how == "mean" else np.nanmedian(arr, axis=1)
    return arr

def parse_version_from_name(path: str):
    """
    Extract semantic version tuple from ..._vX.Y.Z.cdf; returns (X,Y,Z) or (0,0,0) if not found.
    """
    m = re.search(r"_v(\d+)\.(\d+)\.(\d+)\.cdf$", path)
    if not m: return (0,0,0)
    return tuple(int(x) for x in m.groups())

def find_file_for_date(day: datetime) -> Path | None:
    """
    Try to find the file for the given UTC day.
    Looks inside BASE_DIR/<YYYY>/ and picks the latest v*.cdf if multiple.
    """
    y = day.strftime("%Y")
    ymd = day.strftime("%Y%m%d")
    # pattern like: rbspa_rel03_ect-rept-sci-L3_20170527_v*.cdf
    pattern = f"{PROBE}_{RELEASE}_{PRODUCT}_{ymd}_v*.cdf"
    search_dir = Path(BASE_DIR) / y
    candidates = sorted(glob(str(search_dir / pattern)))
    if not candidates:
        print(f"[MISS] No file for {ymd} in {search_dir}")
        return None
    if CHOOSE_LATEST_VERSION:
        candidates.sort(key=parse_version_from_name)  # numeric sort by version
        chosen = candidates[-1]
    else:
        chosen = candidates[0]
    print(f"[HIT] {ymd} → {chosen}")
    return Path(chosen)

def daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ---------- load one file ----------
def load_one(path: Path, prefer_flux=None):
    cdf = cdflib.CDF(str(path))
    flux_name = pick_flux_var(cdf, prefer_flux)
    atts_flux = cdf.varattsget(flux_name)
    fillval = atts_flux.get("FILLVAL", None)
    units_flux_raw = atts_flux.get("UNITS", "")
    units_flux_lbl = istp_units_to_mathtext(units_flux_raw)

    dep0 = atts_flux.get("DEPEND_0")
    dep1 = atts_flux.get("DEPEND_1")
    dep2 = atts_flux.get("DEPEND_2")

    a1 = cdf.varattsget(dep1) if dep1 else {}
    a2 = cdf.varattsget(dep2) if dep2 else {}
    cls1 = classify_axis(dep1, a1) if dep1 else "none"
    cls2 = classify_axis(dep2, a2) if dep2 else "none"
    energy_dep = dep1 if cls1 == "energy" else (dep2 if cls2 == "energy" else None)
    pitch_dep  = dep1 if cls1 == "pitch"  else (dep2 if cls2 == "pitch"  else None)
    if energy_dep is None:
        raise RuntimeError(f"{path.name}: cannot identify energy axis.")

    time = to_dt(cdf.varget(dep0))
    if "L" not in cdf.cdf_info().zVariables:
        raise RuntimeError(f"{path.name}: no 'L' variable found.")
    L = np.squeeze(np.array(cdf.varget("L")))

    energy_vec = np.squeeze(np.array(cdf.varget(energy_dep)))
    energy_units = cdf.varattsget(energy_dep).get("UNITS","")

    flux_raw = cdf.varget(flux_name)  # typically (time, pitch, energy)
    flux = clean_fill(flux_raw, fillval)

    # reorder to (time, pitch, energy) if needed
    if flux.ndim == 3:
        orig_pitch_axis  = 1 if pitch_dep == dep1 else (2 if pitch_dep == dep2 else None)
        orig_energy_axis = 1 if energy_dep == dep1 else (2 if energy_dep == dep2 else None)
        order = [0, orig_pitch_axis, orig_energy_axis]
        if None in order: order = [0,1,2]
        flux = np.moveaxis(flux, [0,1,2], order)
    elif flux.ndim != 2:
        raise RuntimeError(f"{path.name}: unexpected flux dimensionality {flux.ndim}")

    # combine pitch → (time, energy)
    flux_te = combine_pitch(flux, how=PITCH_COMBINE)

    return {
        "flux_name": flux_name,
        "units_flux_lbl": units_flux_lbl,
        "time": time,
        "L": L,
        "energy_vec": energy_vec,
        "energy_units": energy_units,
        "flux_te": flux_te
    }

# ============================ MAIN ============================
if __name__ == "__main__":
    # 1) collect files across date range
    start = datetime.fromisoformat(START_DATE)
    end   = datetime.fromisoformat(END_DATE)
    paths = []
    for day in daterange(start, end):
        p = find_file_for_date(day)
        if p is not None and p.exists():
            paths.append(p)
    if not paths:
        raise SystemExit("[ABORT] No files found in the requested date range.")

    # 2) load all
    parts = []
    for p in paths:
        try:
            parts.append(load_one(p, FLUX_VAR))
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")

    if not parts:
        raise SystemExit("[ABORT] No usable files loaded.")

    # 3) choose energy channels (based on day 1 as reference)
    ref_E = parts[0]["energy_vec"]
    idx_list = [int(np.argmin(np.abs(ref_E - E))) for E in ENERGIES_MEV]
    E_used = [float(ref_E[i]) for i in idx_list]
    print(f"[INFO] Requested energies (MeV): {ENERGIES_MEV}")
    print(f"[INFO] Using nearest channel centers (MeV): {E_used}")
    print(f"[INFO] Channel indices: {idx_list}")

    # 4) concat time/L and build per-energy log10 series
    time_all = np.concatenate([p["time"] for p in parts])
    L_all    = np.concatenate([p["L"] for p in parts])

    log_flux_list = []
    for idx in idx_list:
        series = np.concatenate([p["flux_te"][:, idx] for p in parts]).astype(float)
        series[~np.isfinite(series)] = np.nan
        ok = series > 0
        logf = np.full(series.shape, np.nan)
        logf[ok] = np.log10(series[ok])
        log_flux_list.append(logf)

    flux_name = parts[0]["flux_name"]
    units_flux_lbl = parts[0]["units_flux_lbl"]

    # 5) plot (3×1) with slim colorbar
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, logf, E in zip(axes, log_flux_list, E_used):
        ok = np.isfinite(logf)
        sc = ax.scatter(time_all[ok], L_all[ok], c=logf[ok], s=4,
                        cmap=CMAP, vmin=LOG_VMIN, vmax=LOG_VMAX)
        ax.set_ylabel("L-shell")
        ax.set_title(f"{flux_name}: L vs Time @ ~{E:.2f} MeV (pitch-{PITCH_COMBINE})")

    # keep room for the colorbar on the right
    fig.subplots_adjust(right=0.85, hspace=0.32)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(sc, cax=cbar_ax, label="log$_{10}$ " + (units_flux_lbl or "Flux"))
    axes[-1].set_xlabel("Time (UTC)")

    if SAVE_PLOT:
        fig.savefig(SAVE_NAME, dpi=150)
        print(f"[SAVED] {SAVE_NAME}")

    plt.show()

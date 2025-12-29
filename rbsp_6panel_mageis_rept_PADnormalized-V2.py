#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================== USER SETTINGS ==============================
START_DATE = "2015-06-21"      # inclusive (UTC)
END_DATE   = "2015-06-24"      # inclusive (UTC)

# RBSP file roots (years underneath)
MAG_BASE = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/mageis/level3/pitchangle"
REPT_BASE= "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle"

PROBE     = "rbspa"            # "rbspa" or "rbspb"
MAG_TAG   = "ect-mageis-L3"
REPT_TAG  = "ect-rept-sci-L3"

# Energies for panels (MeV)
E_MAG_TL   = 0.909
E_REPT_TL2 = 2.1
E_REPT_TL3 = 4.2

# Use pitch bin nearest to this angle for normalization reference:
PITCH_SELECT_DEG = 90.0

# Color limits
# Panels 1–3 (non-normalized PAD) in PER keV
LOGR_MAG_TL   = (0.0, 4.0)  # MagEIS (per keV)
LOGR_REPT_TL2 = (-1.0, 3.0)   # REPT (per keV) after conversion from per MeV
LOGR_REPT_TL3 = (-2, 2)   # REPT (per keV) after conversion from per MeV
# Panels 4–6 (normalized PAD): log10(Flux/Flux90), shared range
LOGR_PAD_NORM = (-1.0, 1.0)

# Render non-positive flux (≤0)?
SHOW_NONPOS_AS_FLOOR = True   # True: draw as lowest color; False: leave blank
EPS_FLOOR            = 1e-6   # tiny floor used when True

# Layout
LEFT_MARGIN, RIGHT_MARGIN = 0.10, 0.89
TOP_MARGIN,  BOTTOM_MARGIN = 0.95, 0.12     # a bit more bottom room for the time label
HSPACE = 0.12
ENERGY_LABEL_X = -0.11  # energy text position on left (relative axes coords)

# Output filename (auto from dates)
OUT_NAME = f"rbsp_6panel_mageis_rept_{START_DATE}_{END_DATE}_PADnorm2-V2.png"
# Fonts (bigger)
FS_LABEL = 22
FS_TICK  = 20
FS_CBAR  = 22

# ==========================================================================

import re, sys
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cdflib

# ------------------------------ small utils --------------------------------
def hourly_grid_utc(t0, t1):
    s = pd.Timestamp(t0)
    e = pd.Timestamp(t1)
    s = s.tz_localize("UTC") if s.tz is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tz is None else e.tz_convert("UTC")
    return pd.date_range(start=s.floor("h"), end=e.ceil("h"), freq="1h")

def to_dt_list_any(cdf_epoch_array):
    raw = cdflib.cdfepoch.to_datetime(cdf_epoch_array)
    out = []
    for x in raw:
        ts = pd.Timestamp(x)
        out.append(ts.to_pydatetime().replace(tzinfo=None))
    return np.array(out, dtype=object)

def clean_fill(arr, fillval):
    a = np.array(arr, dtype=float, copy=True)
    if fillval is not None:
        with np.errstate(invalid="ignore"):
            a[a == float(fillval)] = np.nan
    a[np.abs(a) > 1e20] = np.nan
    return a

def pick_flux_var(cdf, prefer=None):
    z = list(cdf.cdf_info().zVariables)
    if prefer in z:
        return prefer
    for name in ("FEDU", "FPDU", "FESA", "FESB", "FESC", "FEO"):
        if name in z:
            return name
    cands = [v for v in z if any(t in v.lower() for t in ("fedu","fpdu","flux","fe"))]
    cands = [v for v in cands if all(b not in v.lower() for b in ("epoch","energy","alpha","delta","labl"))]
    if not cands:
        raise RuntimeError("No flux-like variable found.")
    cands.sort(key=lambda v: np.asarray(cdf.varget(v)).ndim, reverse=True)
    return cands[0]

def classify_axis(varname: str, cdf):
    if not varname:
        return "none"
    atts = cdf.varattsget(varname)
    s = " ".join([
        varname.lower(),
        str(atts.get("UNITS","")).lower(),
        str(atts.get("FIELDNAM","")).lower(),
        str(atts.get("LABLAXIS","")).lower()
    ])
    if any(t in s for t in ("epoch","time","tt2000")): return "time"
    if "energy" in s or any(t in s for t in ("ev","kev","mev","gev")): return "energy"
    if "alpha" in s or "pitch" in s or "deg" in s or "degree" in s:    return "pitch"
    return "unknown"

def parse_version_from_name(path: str):
    m = re.search(r"_v(\d+)\.(\d+)\.(\d+)\.cdf$", path)
    return tuple(map(int, m.groups())) if m else (0,0,0)

def find_daily_file(base_dir: str, ymd: str, probe: str, product_tag: str) -> Path|None:
    folder = Path(base_dir) / ymd[:4]
    pattern = f"{probe}_rel*_{product_tag}_{ymd}_v*.cdf"
    hits = sorted(glob(str(folder / pattern)))
    if not hits:
        print(f"[MISS] {product_tag} {ymd}")
        return None
    hits.sort(key=parse_version_from_name)
    print(f"[HIT]  {product_tag} {ymd} → {hits[-1]}")
    return Path(hits[-1])

def daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def to_log10_with_floor(lin, show_floor=False, eps=1e-6):
    arr = np.asarray(lin, float)
    out = np.full_like(arr, np.nan, float)
    finite = np.isfinite(arr)
    if show_floor:
        out[finite] = np.log10(np.clip(arr[finite], eps, None))
    else:
        pos = finite & (arr > 0)
        out[pos] = np.log10(arr[pos])
    return out

# ----------------------------- CDF day loader -------------------------------
def load_day(path: Path):
    """
    Universal L3 pitch-angle loader (MagEIS or REPT).
    Returns dict:
      time (N,), pitch (P,), energy_MeV (E,),
      flux_tpe_perkeV (N x P x E)  <-- NOTE: converted to per keV for plotting
    """
    cdf = cdflib.CDF(str(path))

    flux_name = pick_flux_var(cdf)
    atts = cdf.varattsget(flux_name)
    fv   = atts.get("FILLVAL", None)
    d0, d1, d2 = atts.get("DEPEND_0"), atts.get("DEPEND_1"), atts.get("DEPEND_2")

    cls1 = classify_axis(d1, cdf)
    cls2 = classify_axis(d2, cdf)
    time_var   = d0
    energy_var = d1 if cls1 == "energy" else (d2 if cls2 == "energy" else None)
    pitch_var  = d1 if cls1 == "pitch"  else (d2 if cls2 == "pitch"  else None)
    if energy_var is None or pitch_var is None:
        raise RuntimeError(f"{path.name}: cannot identify energy AND pitch axes.")

    t = to_dt_list_any(cdf.varget(time_var))
    E = np.squeeze(np.array(cdf.varget(energy_var), dtype=float))
    A = np.squeeze(np.array(cdf.varget(pitch_var),  dtype=float))   # deg

    # energy → MeV (for energy labeling/selection only)
    eunits = str(cdf.varattsget(energy_var).get("UNITS","")).lower()
    if "kev" in eunits:
        energy_MeV = E / 1000.0
    elif "mev" in eunits or eunits == "":
        energy_MeV = E
    elif "ev" in eunits:
        energy_MeV = E / 1e6
    else:
        energy_MeV = E / 1000.0 if E.max() > 200 else E

    # flux → (time, pitch, energy)
    F = clean_fill(cdf.varget(flux_name), fv)
    if F.ndim != 3:
        raise RuntimeError(f"{path.name}: expected 3D flux array (t,pitch,energy).")
    time_axis = np.argmin([abs(s - len(t)) for s in F.shape])
    if time_axis != 0:
        F = np.moveaxis(F, time_axis, 0)
    if F.shape[1] == len(A) and F.shape[2] == len(energy_MeV):
        pass
    elif F.shape[1] == len(energy_MeV) and F.shape[2] == len(A):
        F = np.moveaxis(F, 1, 2)
    else:
        raise RuntimeError(f"{path.name}: cannot align pitch/energy axes.")

    # ---- Convert flux to per keV for plotting in panels 1–3 ----
    units_flux = str(atts.get("UNITS", "")).lower()
    # Correct conversions:
    #   per keV  -> per MeV : multiply by 1000
    #   per MeV  -> per keV : divide by 1000
    # We want per keV here:
    if "kev" in units_flux:
        F_perkeV = F
    elif "mev" in units_flux:
        F_perkeV = F / 1000.0
    elif "ev" in units_flux:
        F_perkeV = F * 1e3  # per eV -> per keV
    else:
        F_perkeV = F  # unknown; assume already per keV

    return dict(time=t, pitch=A, energy_MeV=np.asarray(energy_MeV, float),
                flux_tpe_perkeV=np.asarray(F_perkeV, float))

# ------------------------ extraction & interpolation ------------------------
def nearest_pitch_idx(pitch_vec_deg, target_deg=90.0):
    pv = np.array(pitch_vec_deg, dtype=float).ravel()
    dist = np.minimum(np.abs(pv - target_deg), np.abs((pv % 360.0) - target_deg))
    return int(np.nanargmin(dist))

def time_pitch_at_energy(F_tpe, E_vec, Ereq_mev):
    """Return log10 PAD (non-normalized), using per keV flux."""
    E = np.asarray(E_vec, dtype=float)
    idx = np.where(np.isclose(E, Ereq_mev, atol=1e-6))[0]
    if idx.size:
        M = F_tpe[:, :, idx[0]].astype(float)
    else:
        if not (E.min() <= Ereq_mev <= E.max()):
            raise ValueError(f"Energy {Ereq_mev} MeV outside [{E.min()},{E.max()}] MeV")
        hi = int(np.searchsorted(E, Ereq_mev, side="right")); lo = hi - 1
        w  = (Ereq_mev - E[lo]) / (E[hi] - E[lo])
        f0 = F_tpe[:, :, lo].astype(float); f1 = F_tpe[:, :, hi].astype(float)
        M  = np.full_like(f0, np.nan, float); m  = (f0 > 0) & (f1 > 0)
        M[m] = np.exp(np.log(f0[m]) * (1-w) + np.log(f1[m]) * w)
        left  = (f0 > 0) & ~np.isfinite(f1); right = (f1 > 0) & ~np.isfinite(f0)
        M[left]  = f0[left]; M[right] = f1[right]
    return to_log10_with_floor(M, show_floor=SHOW_NONPOS_AS_FLOOR, eps=EPS_FLOOR)

def time_pitch_at_energy_linear(F_tpe, E_vec, Ereq_mev):
    """Like time_pitch_at_energy(), but returns linear flux (no log)."""
    E = np.asarray(E_vec, dtype=float)
    idx = np.where(np.isclose(E, Ereq_mev, atol=1e-6))[0]
    if idx.size:
        M = F_tpe[:, :, idx[0]].astype(float)
    else:
        if not (E.min() <= Ereq_mev <= E.max()):
            raise ValueError(f"Energy {Ereq_mev} MeV outside [{E.min()},{E.max()}] MeV")
        hi = int(np.searchsorted(E, Ereq_mev, side="right")); lo = hi - 1
        w  = (Ereq_mev - E[lo]) / (E[hi] - E[lo])
        f0 = F_tpe[:, :, lo].astype(float); f1 = F_tpe[:, :, hi].astype(float)
        M  = np.full_like(f0, np.nan, float); m  = (f0 > 0) & (f1 > 0)
        M[m] = np.exp(np.log(f0[m]) * (1-w) + np.log(f1[m]) * w)
        left  = (f0 > 0) & ~np.isfinite(f1); right = (f1 > 0) & ~np.isfinite(f0)
        M[left]  = f0[left]; M[right] = f1[right]
    return M  # linear per keV

def norm_pad_log(F_tpe, E_vec, Ereq_mev, pitch_vec, psel_deg=90.0):
    """Return log10(Flux/Flux90) PAD with a floor to avoid white gaps (unitless)."""
    M_lin = time_pitch_at_energy_linear(F_tpe, E_vec, Ereq_mev)
    pidx  = nearest_pitch_idx(pitch_vec, psel_deg)
    ref   = M_lin[:, pidx]                                  # (time,)
    norm  = M_lin / np.maximum(ref[:, None], EPS_FLOOR)     # safe normalization
    return to_log10_with_floor(norm, show_floor=True, eps=EPS_FLOOR)

# ----------------------------------- MAIN -----------------------------------
if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
    })

    start = datetime.fromisoformat(START_DATE)
    end   = datetime.fromisoformat(END_DATE)

    # ---------- MagEIS ----------
    mag_paths = []
    for d in daterange(start, end):
        ymd = d.strftime("%Y%m%d")
        p = find_daily_file(MAG_BASE, ymd, PROBE, MAG_TAG)
        if p and p.exists(): mag_paths.append(p)
    if not mag_paths: sys.exit("[ABORT] No MagEIS files found.")
    mag_parts = [load_day(p) for p in mag_paths]
    mag_time  = np.concatenate([x["time"] for x in mag_parts])
    mag_E     = mag_parts[0]["energy_MeV"]
    mag_A     = mag_parts[0]["pitch"]
    mag_F     = np.concatenate([x["flux_tpe_perkeV"] for x in mag_parts], axis=0)  # per keV

    # ---------- REPT ----------
    rept_paths = []
    for d in daterange(start, end):
        ymd = d.strftime("%Y%m%d")
        p = find_daily_file(REPT_BASE, ymd, PROBE, REPT_TAG)
        if p and p.exists(): rept_paths.append(p)
    if not rept_paths: sys.exit("[ABORT] No REPT files found.")
    rept_parts = [load_day(p) for p in rept_paths]
    rept_time  = np.concatenate([x["time"] for x in rept_parts])
    rept_E     = rept_parts[0]["energy_MeV"]
    rept_A     = rept_parts[0]["pitch"]
    rept_F     = np.concatenate([x["flux_tpe_perkeV"] for x in rept_parts], axis=0)  # converted to per keV

    # -------------------------- Build PAD matrices ---------------------------
    # Non-normalized PADs (log10 flux per keV)
    mag_pad_log   = time_pitch_at_energy(mag_F,   mag_E,   E_MAG_TL)
    rept2_pad_log = time_pitch_at_energy(rept_F,  rept_E,  E_REPT_TL2)
    rept3_pad_log = time_pitch_at_energy(rept_F,  rept_E,  E_REPT_TL3)

    # Normalized PADs: log10(Flux/Flux90) (unitless)
    mag_pad_norm_log   = norm_pad_log(mag_F,  mag_E,  E_MAG_TL,   mag_A,  PITCH_SELECT_DEG)
    rept2_pad_norm_log = norm_pad_log(rept_F, rept_E, E_REPT_TL2, rept_A, PITCH_SELECT_DEG)
    rept3_pad_norm_log = norm_pad_log(rept_F, rept_E, E_REPT_TL3, rept_A, PITCH_SELECT_DEG)

    # ------------------------------ PLOT -------------------------------------
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)  # share x for easy tick management
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        top=TOP_MARGIN, bottom=BOTTOM_MARGIN, hspace=HSPACE)

    extend_kw = 'min' if SHOW_NONPOS_AS_FLOOR else 'neither'
    CBAR_LABEL_FLUX_KEV = r"log$_{10}$(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)"
    CBAR_LABEL_NORM     = r"log$_{10}$(Flux / Flux$_{90^\circ}$)"

    # Time arrays
    t_mag  = pd.to_datetime(mag_time,  utc=True)
    t_rept = pd.to_datetime(rept_time, utc=True)
    xmin = min(t_mag.min(), t_rept.min())
    xmax = max(t_mag.max(), t_rept.max())

    # -------- Panels 1–3: non-normalized PADs (log10 flux per keV) ----------
    # P1: MagEIS 0.909 MeV
    ax = axes[0]
    pm1 = ax.pcolormesh(t_mag, mag_A, mag_pad_log.T, shading="nearest", cmap="jet",
                        vmin=LOGR_MAG_TL[0], vmax=LOGR_MAG_TL[1])
    ax.set_ylabel("Pitch (deg)")
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_MAG_TL:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # P2: REPT 2.1 MeV
    ax = axes[1]
    pm2 = ax.pcolormesh(t_rept, rept_A, rept2_pad_log.T, shading="nearest", cmap="jet",
                        vmin=LOGR_REPT_TL2[0], vmax=LOGR_REPT_TL2[1])
    ax.set_ylabel("Pitch (deg)")
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL2:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # P3: REPT 4.2 MeV
    ax = axes[2]
    pm3 = ax.pcolormesh(t_rept, rept_A, rept3_pad_log.T, shading="nearest", cmap="jet",
                        vmin=LOGR_REPT_TL3[0], vmax=LOGR_REPT_TL3[1])
    ax.set_ylabel("Pitch (deg)")
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL3:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # Add individual colorbars for 1–3 (ranges differ), but only ONE units label:
    cb1 = fig.colorbar(pm1, ax=axes[0], pad=0.01, fraction=0.035, extend=extend_kw)
    cb2 = fig.colorbar(pm2, ax=axes[1], pad=0.01, fraction=0.035, extend=extend_kw)
    cb3 = fig.colorbar(pm3, ax=axes[2], pad=0.01, fraction=0.035, extend=extend_kw)
    for cb in (cb1, cb2, cb3):
        cb.ax.tick_params(labelsize=FS_TICK)
    # One shared label for panels 1–3:
    fig.text(0.93, 0.72, CBAR_LABEL_FLUX_KEV, rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # -------- Panels 4–6: normalized PADs log10(Flux/Flux90) -----------------
    VLO, VHI = LOGR_PAD_NORM

    ax4 = axes[3]
    pm4 = ax4.pcolormesh(t_mag, mag_A, mag_pad_norm_log.T, shading="nearest", cmap="jet",
                         vmin=VLO, vmax=VHI)
    ax4.set_ylabel("Pitch (deg)")
    ax4.text(ENERGY_LABEL_X, 0.5, f"{E_MAG_TL:g} MeV",
             transform=ax4.transAxes, rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    ax5 = axes[4]
    pm5 = ax5.pcolormesh(t_rept, rept_A, rept2_pad_norm_log.T, shading="nearest", cmap="jet",
                         vmin=VLO, vmax=VHI)
    ax5.set_ylabel("Pitch (deg)")
    ax5.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL2:g} MeV",
             transform=ax5.transAxes, rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    ax6 = axes[5]
    pm6 = ax6.pcolormesh(t_rept, rept_A, rept3_pad_norm_log.T, shading="nearest", cmap="jet",
                         vmin=VLO, vmax=VHI)
    ax6.set_ylabel("Pitch (deg)")
    ax6.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL3:g} MeV",
             transform=ax6.transAxes, rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # One shared colorbar for 4–6:
    cb_norm = fig.colorbar(pm6, ax=[axes[3], axes[4], axes[5]],
                           pad=0.01, fraction=0.035, extend=extend_kw)
    cb_norm.ax.tick_params(labelsize=FS_TICK)
    cb_norm.set_label(r"log$_{10}$(Flux / Flux$_{90^\circ}$)", fontsize=FS_CBAR)

    # -------- X-axis formatting --------
    # Only panel 6 keeps tick labels + x-label; others cancelled
    for ax in axes[:-1]:
        ax.tick_params(axis='x', which='both', labelbottom=False)
    axes[-1].tick_params(axis='x', which='both', labelbottom=True)
    # Consistent time range
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.margins(x=0)
    # Sensible ticks & label on the last panel only
    locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator, tz=timezone.utc)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(formatter)
    axes[-1].set_xlabel("Time (UTC)", fontsize=FS_LABEL, labelpad=8)

    fig.savefig(OUT_NAME, dpi=180, bbox_inches="tight", pad_inches=0.02)
    print(f"[SAVED] {OUT_NAME}")
    plt.show()

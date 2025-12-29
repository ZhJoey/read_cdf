#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================== USER SETTINGS ==============================
START_DATE = "2017-05-27"      # inclusive (UTC)
END_DATE   = "2017-05-28"      # inclusive (UTC)

# RBSP file roots (years underneath)
MAG_BASE = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/mageis/level3/pitchangle"
REPT_BASE= "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle"

PROBE     = "rbspa"            # "rbspa" or "rbspb"
MAG_TAG   = "ect-mageis-L3"
REPT_TAG  = "ect-rept-sci-L3"

# Energy for both panels (MeV)
E_REPT = 2.1

# Pitch-angle to use as normalization reference:
PITCH_SELECT_DEG = 90.0

# Color limits
# Panel 1 (PAD) — log10 flux per keV
LOGR_PAD = (-1.0, 3.0)
# Panel 2 (normalized PAD)
# NORM_USE_LOG   = True
NORM_USE_LOG   = False               # <— set False to see linear ratio (no log)
LOGR_PAD_NORM  = (-1.0, 1.0)       # used if NORM_USE_LOG=True
LINR_PAD_NORM  = (0.1, 2.1)        # used if NORM_USE_LOG=False

# Render non-positive flux (≤0)?
SHOW_NONPOS_AS_FLOOR = True
EPS_FLOOR            = 1e-6        # tiny floor used when True

# Layout
LEFT_MARGIN, RIGHT_MARGIN = 0.05, 0.85
TOP_MARGIN,  BOTTOM_MARGIN = 0.93, 0.14
HSPACE = 0.12
ENERGY_LABEL_X = -0.11

# Output filename (auto from dates)
OUT_NAME = f"/home/zhouyu/read_cdf/plot_prague/result/rbsp_2panel_rept_{START_DATE}_{END_DATE}_2p1MeV_perkeV_normPAD{'_log' if NORM_USE_LOG else '_linear'}_2.png"

# Fonts
FS_LABEL = 24
FS_TICK  = 24
FS_CBAR  = 24

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
      flux_tpe_perkeV (N x P x E)  <-- converted to per keV
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
        energy_MeV = E / 1000.0; scale = "keV"
    elif "mev" in eunits or eunits == "":
        energy_MeV = E;           scale = "mev"
    elif "ev" in eunits:
        energy_MeV = E / 1e6;     scale = "ev"
    else:
        energy_MeV = E / 1000.0 if E.max() > 200 else E; scale = "auto"

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

    # ---- Convert flux to per keV (careful and explicit) --------------------
    units_flux = (atts.get("UNITS", "") or "").lower()
    if "kev" in units_flux:
        F_perkeV = F
    elif "mev" in units_flux:
        F_perkeV = F / 1000.0          # per MeV → per keV
    elif re.search(r"\bev\b", units_flux):
        F_perkeV = F * 1000.0          # per eV → per keV
    else:
        # fallback by inferred energy scale if units missing
        if scale == "keV":
            F_perkeV = F
        elif scale == "mev":
            F_perkeV = F / 1000.0
        elif scale == "ev":
            F_perkeV = F * 1000.0
        else:
            F_perkeV = F

    return dict(time=t, pitch=A, energy_MeV=np.asarray(energy_MeV, float),
                flux_tpe_perkeV=np.asarray(F_perkeV, float))

# ------------------------ extraction & interpolation ------------------------
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
    """Same as above but returns linear flux (per keV)."""
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

def nearest_pitch_idx(pitch_vec_deg, target_deg=90.0):
    pv = np.array(pitch_vec_deg, dtype=float).ravel()
    dist = np.minimum(np.abs(pv - target_deg), np.abs((pv % 360.0) - target_deg))
    return int(np.nanargmin(dist))

def norm_pad_linear(F_tpe, E_vec, Ereq_mev, pitch_vec, psel_deg=90.0):
    """Return linear Flux/Flux90 PAD (unitless)."""
    M_lin = time_pitch_at_energy_linear(F_tpe, E_vec, Ereq_mev)
    pidx  = nearest_pitch_idx(pitch_vec, psel_deg)
    ref   = M_lin[:, pidx]                               # (time,)
    norm  = M_lin / np.maximum(ref[:, None], EPS_FLOOR)  # safe normalization
    norm[~np.isfinite(M_lin)] = np.nan
    return norm

def norm_pad_log(F_tpe, E_vec, Ereq_mev, pitch_vec, psel_deg=90.0):
    """Return log10(Flux/Flux90) PAD (unitless)."""
    return to_log10_with_floor(norm_pad_linear(F_tpe, E_vec, Ereq_mev, pitch_vec, psel_deg),
                               show_floor=True, eps=EPS_FLOOR)

# ----------------------------------- MAIN -----------------------------------
if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "font.weight": "bold",
        "axes.labelweight": "bold",
    })

    start = datetime.fromisoformat(START_DATE)
    end   = datetime.fromisoformat(END_DATE)

    # ---------- REPT only (2.1 MeV) ----------
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
    rept_F     = np.concatenate([x["flux_tpe_perkeV"] for x in rept_parts], axis=0)  # per keV

    # -------------------------- Build PAD matrices ---------------------------
    # Panel 1: non-normalized PAD (log10 per keV)
    rept_pad_log = time_pitch_at_energy(rept_F, rept_E, E_REPT)

    # Panel 2: normalized PAD (linear or log depending on NORM_USE_LOG)
    if NORM_USE_LOG:
        rept_pad_norm = norm_pad_log(rept_F, rept_E, E_REPT, rept_A, PITCH_SELECT_DEG)
        VMIN, VMAX = LOGR_PAD_NORM
        NORM_LABEL = r"log$_{10}$(Flux / Flux$_{90^\circ}$)"
    else:
        rept_pad_norm = norm_pad_linear(rept_F, rept_E, E_REPT, rept_A, PITCH_SELECT_DEG)
        VMIN, VMAX = LINR_PAD_NORM
        NORM_LABEL = r"Flux / Flux$_{90^\circ}$"

    # ------------------------------ PLOT -------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        top=TOP_MARGIN, bottom=BOTTOM_MARGIN, hspace=HSPACE)
    # fig.suptitle(f"{PROBE.upper()} REPT — 2.1 MeV (per keV units); PAD & normalized PAD", fontsize=FS_LABEL+2)

    extend_kw = 'min' if SHOW_NONPOS_AS_FLOOR else 'neither'
    CBAR_LABEL_FLUX_KEV = r"log$_{10}$(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)"

    # Time range
    t_rept = pd.to_datetime(rept_time, utc=True)
    xmin, xmax = t_rept.min(), t_rept.max()

    # Panel 1
    ax1 = axes[0]
    pm1 = ax1.pcolormesh(t_rept, rept_A, rept_pad_log.T, shading="nearest", cmap="jet",
                         vmin=LOGR_PAD[0], vmax=LOGR_PAD[1])
    ax1.set_ylabel("Pitch (deg)")
    # ax1.text(ENERGY_LABEL_X, 0.5, f"{E_REPT:g} MeV", transform=ax1.transAxes,
    #          rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    cb1 = fig.colorbar(pm1, ax=ax1, pad=0.01, fraction=0.04, extend=extend_kw)
    cb1.ax.tick_params(labelsize=FS_TICK)
    fig.text(0.93, 0.73, CBAR_LABEL_FLUX_KEV, rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # Panel 2
    ax2 = axes[1]
    pm2 = ax2.pcolormesh(t_rept, rept_A, rept_pad_norm.T, shading="nearest", cmap="jet",
                         vmin=VMIN, vmax=VMAX)
    ax2.set_ylabel("Pitch (deg)")
    # ax2.text(ENERGY_LABEL_X, 0.5, f"{E_REPT:g} MeV", transform=ax2.transAxes,
    #          rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    cb2 = fig.colorbar(pm2, ax=ax2, pad=0.01, fraction=0.04)
    cb2.ax.tick_params(labelsize=FS_TICK)
    cb2.set_label(NORM_LABEL, fontsize=FS_CBAR)

    # X-axis formatting on bottom panel
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.margins(x=0)
    # ticks every 12 hours, label on two lines: "May 27" newline "00:00"
    # Force ticks at 00:00 and 12:00 every day, including first and last day
    locator = mdates.HourLocator(byhour=[0, 12], tz=timezone.utc)
    formatter = mdates.DateFormatter("%b %d\n%H:%M", tz=timezone.utc)

    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    # Force extra ticks at min/max time boundaries
    ticks = pd.date_range(start=xmin.floor("D"), end=xmax.ceil("D"),
                        freq="12H", tz="UTC")
    ax2.set_xticks(ticks)
    ax2.xaxis.set_major_formatter(formatter)



    fig.savefig(OUT_NAME, dpi=180, bbox_inches="tight", pad_inches=0.02)
    print(f"[SAVED] {OUT_NAME}")
    plt.show()

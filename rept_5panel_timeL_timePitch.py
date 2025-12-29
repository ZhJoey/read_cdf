#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBSP/REPT 5-panel plot
Panels:
 1) Time–L @ 1.8 MeV     (pitch bin ≈ 90°)
 2) Time–L @ 2.1 MeV     (pitch bin ≈ 90°)
 3) Time–L @ 4.2 MeV     (pitch bin ≈ 90°)
 4) Time–Pitch @ 2.1 MeV
 5) Kp (right axis) & Dst (left axis)

Notes:
- Flux is shown as log10(cm^-2 s^-1 sr^-1 MeV^-1)
- Energy selection uses nearest channel; if not exact, uses log-linear interpolation
- Time–L panels use the single pitch bin nearest PITCH_SELECT_DEG (default 90°)
"""

# ============================== USER SETTINGS ==============================
START_DATE = "2015-06-21"          # inclusive, UTC
END_DATE   = "2015-06-24"          # inclusive, UTC

BASE_DIR = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle"
PROBE    = "rbspa"                 # "rbspa" or "rbspb"
RELEASE  = "rel03"
PRODUCT  = "ect-rept-sci-L3"
CHOOSE_LATEST_VERSION = True

FLUX_VAR = None                    # None=auto (prefers "FEDU", else "FPDU")
PITCH_SELECT_DEG = 90.0            # nearest pitch bin used for Time–L panels
TP_ENERGY = 2.1                    # MeV for the Time–Pitch panel

# Energies (MeV) for Time–L panels (in this exact order)
TL_ENERGIES = [1.8, 2.1, 4.2]      # MeV

# Color limits (log10 flux per MeV). Provide one (vmin, vmax) per TL energy.
TL_LOG_RANGES = [(2, 6), (2, 6), (1, 4)]
# Time–Pitch panel color range (log10)
TP_LOG_RANGE = (2, 6)

# Compact layout tweaks
LEFT_MARGIN, RIGHT_MARGIN = 0.12, 0.88
TOP_MARGIN,  BOTTOM_MARGIN = 0.95, 0.09
HSPACE = 0.16
ENERGY_LABEL_X = -0.10            # left-side vertical energy text (axes coords)

SAVE_PNG = True
OUT_NAME = "rept_5panel_timeL_timePitch.png"

# Fonts
FS_SUP   = 18
FS_LABEL = 15
FS_TICK  = 12
FS_CBAR  = 13
# ==========================================================================

import re
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import cdflib

# ------------------------------ helpers -----------------------------------
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
    for name in ("FEDU", "FPDU"):
        if name in z:
            return name
    cands = [v for v in z if any(t in v.lower() for t in ("flux","fedu","fpdu"))]
    cands = [v for v in cands if all(b not in v.lower() for b in ("epoch","energy","alpha","delta","labl"))]
    if not cands:
        raise RuntimeError("No flux-like variable found. Set FLUX_VAR.")
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

def find_daily_file(day: datetime) -> Path | None:
    y = day.strftime("%Y"); ymd = day.strftime("%Y%m%d")
    pattern = f"{PROBE}_{RELEASE}_{PRODUCT}_{ymd}_v*.cdf"
    folder = Path(BASE_DIR) / y
    hits = sorted(glob(str(folder / pattern)))
    if not hits:
        print(f"[MISS] {ymd} (no file)")
        return None
    if CHOOSE_LATEST_VERSION:
        hits.sort(key=parse_version_from_name)
        chosen = hits[-1]
    else:
        chosen = hits[0]
    print(f"[HIT]  {ymd} → {chosen}")
    return Path(chosen)

def daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ------------------------------ OMNI (Kp/Dst) ------------------------------
def _fetch_omni_series(start_dt, end_dt, varnum):
    base = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    params = {
        "activity": "ftp", "res": "hour", "spacecraft": "omni2",
        "start_date": start_dt.strftime("%Y%m%d"),
        "end_date":   (end_dt - timedelta(days=0)).strftime("%Y%m%d"),
        "maxdays": str((end_dt - start_dt).days + 1),
        "vars": varnum, "view": 0, "nsum": 1, "paper": 0, "table": 0
    }
    r = requests.get(base, params=params, timeout=30); r.raise_for_status()
    m = re.search(r"(https?://omniweb\.gsfc\.nasa\.gov/staging/omni2_[\w\-]+\.lst)", r.text)
    if not m:
        raise RuntimeError("OMNI staging link not found.")
    lst_url = m.group(1)
    r2 = requests.get(lst_url, timeout=30); r2.raise_for_status()

    rows = []
    for line in r2.text.splitlines():
        s = line.strip()
        if not s or s[0] in "#;": continue
        parts = s.split()
        try:
            if len(parts) >= 4 and len(parts[1]) <= 3:     # YYYY DOY HH VAL
                yyyy, doy, hh = int(parts[0]), int(parts[1]), int(parts[2])
                val = float(parts[3])
                dt = datetime(yyyy, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy-1, hours=hh)
            elif len(parts) >= 5:                          # YYYY MM DD HH VAL
                yyyy, mm, dd, hh = map(int, parts[:4])
                val = float(parts[4])
                dt = datetime(yyyy, mm, dd, hh, tzinfo=timezone.utc)
            else:
                continue
            rows.append((dt, val))
        except Exception:
            continue

    if not rows:
        raise RuntimeError("OMNI parsed no values.")
    idx = pd.DatetimeIndex([t for t,_ in rows], tz="UTC")
    vals = np.array([v for _,v in rows], dtype=float)
    return pd.Series(vals, index=idx).sort_index()

def fetch_kp(start_dt, end_dt):
    s = _fetch_omni_series(start_dt, end_dt, 38)  # Kp*10
    hourly = pd.date_range(start=s.index.min(), end=s.index.max(), freq="1h", tz="UTC")
    return pd.DataFrame({"Kp": (s/10.0).reindex(hourly, method="ffill")})

def fetch_dst(start_dt, end_dt):
    s = _fetch_omni_series(start_dt, end_dt, 40)  # Dst
    hourly = pd.date_range(start=s.index.min(), end=s.index.max(), freq="1h", tz="UTC")
    return pd.DataFrame({"Dst": s.reindex(hourly, method="ffill")})

# ------------------------------- CDF loader --------------------------------
def load_rept_day(path: Path):
    """
    Returns dict with:
      time (N,), L (N,), pitch_vec (P,), energy_vec (E,),
      flux_tpe (N x P x E), flux_units (str)
    """
    cdf = cdflib.CDF(str(path))

    # Flux variable and dependencies
    flux_name = pick_flux_var(cdf, FLUX_VAR)
    atts = cdf.varattsget(flux_name)
    dep0, dep1, dep2 = atts.get("DEPEND_0"), atts.get("DEPEND_1"), atts.get("DEPEND_2")
    fv = atts.get("FILLVAL", None)

    # Identify axes
    cls1 = classify_axis(dep1, cdf)
    cls2 = classify_axis(dep2, cdf)
    time_var = dep0
    energy_var = dep1 if cls1 == "energy" else (dep2 if cls2 == "energy" else None)
    pitch_var  = dep1 if cls1 == "pitch"  else (dep2 if cls2 == "pitch"  else None)
    if energy_var is None or pitch_var is None:
        raise RuntimeError("Cannot find both pitch and energy axes; need pitch-angle product.")

    # Vectors
    t = to_dt_list_any(cdf.varget(time_var))
    E = np.squeeze(np.array(cdf.varget(energy_var), dtype=float))  # MeV
    A = np.squeeze(np.array(cdf.varget(pitch_var),  dtype=float))  # deg

    # L (or L_star)
    zvars = cdf.cdf_info().zVariables
    if "L" in zvars:
        L = clean_fill(cdf.varget("L"), cdf.varattsget("L").get("FILLVAL", None))
    elif "L_star" in zvars:
        L = clean_fill(cdf.varget("L_star"), cdf.varattsget("L_star").get("FILLVAL", None))
    else:
        L = np.full(len(t), np.nan)
    L = np.squeeze(L).astype(float)
    with np.errstate(invalid="ignore"):
        L[(L < 0.5) | (L > 10.0)] = np.nan

    # Flux array → reorder to (time, pitch, energy)
    F = clean_fill(cdf.varget(flux_name), fv)
    # find time axis
    if F.ndim != 3:
        raise RuntimeError("Expected a 3D flux array (time, pitch, energy).")
    time_axis = np.argmin([abs(s - len(t)) for s in F.shape])
    if time_axis != 0:
        F = np.moveaxis(F, time_axis, 0)

    # remaining axes must be pitch & energy; guess by matching lengths
    if F.shape[1] == len(A) and F.shape[2] == len(E):
        pass
    elif F.shape[1] == len(E) and F.shape[2] == len(A):
        F = np.moveaxis(F, 1, 2)  # swap to (..., pitch, energy)
    else:
        raise RuntimeError("Cannot align pitch/energy axes.")

    # Units (pretty text)
    units = atts.get("UNITS", "cm^-2 s^-1 sr^-1 MeV^-1")

    return dict(time=t, L=L, pitch=A, energy=E, flux_tpe=F, units=units, flux_name=flux_name)

# ------------------------ extraction & interpolation -----------------------
def pick_pitch_index(pitch_vec_deg, target_deg=90.0):
    pv = np.array(pitch_vec_deg, dtype=float).ravel()
    dist = np.minimum(np.abs(pv - target_deg), np.abs((pv % 360.0) - target_deg))
    return int(np.nanargmin(dist))

def series_timeL_at_energy(F_tpe, E_vec, pitch_vec, energy_req_mev, pitch_deg=90.0,
                           exact_only=False):
    """Return (log10 flux per MeV at time, how_str) for Time–L panels."""
    E = np.asarray(E_vec, dtype=float)
    # energy index or interpolate
    idx = np.where(np.isclose(E, energy_req_mev, atol=1e-6))[0]
    if idx.size:
        e_idx = idx[0]
        F_te = F_tpe[:, pick_pitch_index(pitch_vec, pitch_deg), e_idx]
        how = "exact"
    else:
        if not (E.min() <= energy_req_mev <= E.max()):
            raise ValueError(f"E={energy_req_mev} MeV outside [{E.min()}, {E.max()}] MeV")
        if exact_only:
            raise ValueError(f"E={energy_req_mev} MeV not an exact channel.")
        hi = int(np.searchsorted(E, energy_req_mev, side="right"))
        lo = hi - 1
        w  = (energy_req_mev - E[lo]) / (E[hi] - E[lo])
        f0 = F_tpe[:, pick_pitch_index(pitch_vec, pitch_deg), lo]
        f1 = F_tpe[:, pick_pitch_index(pitch_vec, pitch_deg), hi]
        out = np.full_like(f0, np.nan, dtype=float)
        m = (f0 > 0) & (f1 > 0)
        out[m] = np.exp(np.log(f0[m]) * (1-w) + np.log(f1[m]) * w)
        out[(f0 > 0) & ~np.isfinite(f1)] = f0[(f0 > 0) & ~np.isfinite(f1)]
        out[(f1 > 0) & ~np.isfinite(f0)] = f1[(f1 > 0) & ~np.isfinite(f0)]
        F_te = out
        how = f"interp {E[lo]:g}-{E[hi]:g}"
    # log10
    logf = np.full_like(F_te, np.nan, dtype=float)
    ok = F_te > 0
    logf[ok] = np.log10(F_te[ok])
    return logf, how

def map_time_pitch_at_energy(F_tpe, E_vec, energy_req_mev):
    """Return 2D array (time x pitch) at the requested energy (log10)."""
    E = np.asarray(E_vec, dtype=float)
    idx = np.where(np.isclose(E, energy_req_mev, atol=1e-6))[0]
    if idx.size:
        M = F_tpe[:, :, idx[0]].astype(float)
    else:
        if not (E.min() <= energy_req_mev <= E.max()):
            raise ValueError(f"E={energy_req_mev} MeV outside [{E.min()}, {E.max()}] MeV")
        hi = int(np.searchsorted(E, energy_req_mev, side="right"))
        lo = hi - 1
        w  = (energy_req_mev - E[lo]) / (E[hi] - E[lo])
        f0 = F_tpe[:, :, lo].astype(float)
        f1 = F_tpe[:, :, hi].astype(float)
        M  = np.full_like(f0, np.nan, dtype=float)
        m = (f0 > 0) & (f1 > 0)
        M[m] = np.exp(np.log(f0[m]) * (1-w) + np.log(f1[m]) * w)
        left  = (f0 > 0) & ~np.isfinite(f1)
        right = (f1 > 0) & ~np.isfinite(f0)
        M[left]  = f0[left]
        M[right] = f1[right]
    # log10
    out = np.full_like(M, np.nan, dtype=float)
    ok = M > 0
    out[ok] = np.log10(M[ok])
    return out

# ----------------------------------- MAIN ----------------------------------
if __name__ == "__main__":
    # Fonts
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
    })

    # Gather files
    start = datetime.fromisoformat(START_DATE)
    end   = datetime.fromisoformat(END_DATE)
    paths = []
    for d in daterange(start, end):
        p = find_daily_file(d)
        if p and p.exists():
            paths.append(p)
    if not paths:
        raise SystemExit("[ABORT] No REPT files found for the requested dates.")
    print(f"[info] Using {len(paths)} daily file(s).")

    # Load & concat days
    parts = [load_rept_day(p) for p in paths]
    time = np.concatenate([p["time"] for p in parts])
    L    = np.concatenate([p["L"]    for p in parts])
    # assume same pitch/energy grids
    pitch = parts[0]["pitch"]
    energy = parts[0]["energy"]
    F_list = [p["flux_tpe"] for p in parts]
    F = np.concatenate(F_list, axis=0)  # (time x pitch x energy)

    # Build the 3 time–L series (log10)
    tl_series = []
    how_list  = []
    for Ereq, _rng in zip(TL_ENERGIES, TL_LOG_RANGES):
        s, how = series_timeL_at_energy(F, energy, pitch, Ereq, pitch_deg=PITCH_SELECT_DEG)
        tl_series.append(s); how_list.append(how)
        print(f"[info] Time–L @ {Ereq} MeV → {how}")

    # Build time–pitch map @ TP_ENERGY (log10)
    tp_map = map_time_pitch_at_energy(F, energy, TP_ENERGY)
    print(f"[info] Time–Pitch @ {TP_ENERGY} MeV ready.")

    # Kp/Dst
    x_time = pd.to_datetime(time, utc=True)
    xmin, xmax = x_time.min(), x_time.max()
    start_dt = xmin.floor("h").to_pydatetime()
    end_dt   = (xmax.ceil("h") + pd.Timedelta(hours=1)).to_pydatetime()

    kp_df = dst_df = None
    try:
        kp_df = fetch_kp(start_dt, end_dt);  print(f"[ok] Kp points: {len(kp_df)}")
    except Exception as e:
        print(f"[warn] Kp download failed: {e}")
    try:
        dst_df = fetch_dst(start_dt, end_dt); print(f"[ok] Dst points: {len(dst_df)}")
    except Exception as e:
        print(f"[warn] Dst download failed: {e}")

    # ------------------------------ PLOT --------------------------------
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True,
                             gridspec_kw={"height_ratios":[1,1,1,1.1,0.7]})
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        top=TOP_MARGIN, bottom=BOTTOM_MARGIN, hspace=HSPACE)
    fig.suptitle(
        f"{PROBE.upper()}  Time–L (three energies) & Time–Pitch (one energy) + Kp/Dst\n"
        f"{START_DATE} to {END_DATE} (UTC)   |   pitch≈{PITCH_SELECT_DEG:.0f}° for Time–L",
        fontsize=FS_SUP
    )

    # Panels 1–3: Time–L (scatter colored by log10 flux)
    for i, (ax, logf, Ereq, (vmin, vmax)) in enumerate(
        zip(axes[:3], tl_series, TL_ENERGIES, TL_LOG_RANGES)
    ):
        ok = np.isfinite(logf) & np.isfinite(L)
        sc = ax.scatter(x_time[ok], L[ok], c=logf[ok], s=6, cmap="jet",
                        vmin=vmin, vmax=vmax)
        # L limits w/ small headroom
        if np.any(ok):
            lo, hi = np.nanpercentile(L[ok], [1, 99])
            lo = max(0.8, lo)
            hi = min(10.0, hi + 0.15*(hi-lo))
            ax.set_ylim(lo, hi)
        else:
            ax.set_ylim(1.0, 7.0)
        ax.set_ylabel("L")
        ax.set_xlim(xmin, xmax); ax.margins(x=0)
        # vertical energy label on the left
        ax.text(ENERGY_LABEL_X, 0.5, f"{Ereq:g} MeV", transform=ax.transAxes,
                rotation=90, va="center", ha="right", fontsize=FS_LABEL)
        # per-panel colorbar (no text—same units for all)
        cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04)
        cb.ax.tick_params(labelsize=FS_TICK)

    # ONE shared colorbar label on the right
    fig.text(0.93, 0.53, r"log$_{10}$(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ MeV$^{-1}$)",
             rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # Panel 4: Time–Pitch @ TP_ENERGY
    ax4 = axes[3]
    pm = ax4.pcolormesh(
        x_time, pitch, tp_map.T, shading="nearest", cmap="jet",
        vmin=TP_LOG_RANGE[0], vmax=TP_LOG_RANGE[1]
    )
    ax4.set_ylabel("Pitch (deg)")
    ax4.set_xlim(xmin, xmax); ax4.margins(x=0)
    cb4 = fig.colorbar(pm, ax=ax4, pad=0.01, fraction=0.04)
    cb4.ax.tick_params(labelsize=FS_TICK)
    # left label with energy (vertical)
    ax4.text(ENERGY_LABEL_X, 0.5, f"{TP_ENERGY:g} MeV", transform=ax4.transAxes,
             rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # Panel 5: Dst (left) and Kp (right)
    ax5 = axes[4]
    ax5.grid(True, alpha=0.3, linestyle=":")
    ax5.set_ylabel("Dst (nT)")
    hours = pd.date_range(start=xmin.floor("h"), end=xmax.ceil("h"), freq="1h", tz="UTC")

    if dst_df is not None and not dst_df.empty:
        dst_df.index = pd.to_datetime(dst_df.index, utc=True)
        dst_plot = dst_df.reindex(hours).interpolate(limit_direction="both")
        ax5.plot(dst_plot.index, dst_plot["Dst"], lw=1.5, color="k", label="Dst")

    if kp_df is not None and not kp_df.empty:
        kp_df.index = pd.to_datetime(kp_df.index, utc=True)
        kp_plot = kp_df.reindex(hours).interpolate(limit_direction="both")
        ax5r = ax5.twinx()
        ax5r.step(kp_plot.index, kp_plot["Kp"], where="post",
                  lw=1.5, color="tab:orange", label="Kp")
        ax5r.set_ylabel("Kp", color="tab:orange")
        ax5r.tick_params(axis='y', labelcolor="tab:orange")
        ax5r.set_ylim(0, 9)
        # merge legends
        lines, labels = [], []
        for axx in (ax5, ax5r):
            h, l = axx.get_legend_handles_labels()
            lines += h; labels += l
        ax5.legend(lines, labels, loc="upper right")

    axes[-1].set_xlabel("Time (UTC)")
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H", tz=timezone.utc))

    if SAVE_PNG:
        fig.savefig(OUT_NAME, dpi=180, bbox_inches="tight", pad_inches=0.02)
        print(f"[SAVED] {OUT_NAME}")

    plt.show()

    # Assume you already built tp_map USING LINEAR (not yet log) values at 2.1 MeV:
    E = 2.1
    e_idx = np.where(np.isclose(energy, E, atol=1e-6))[0]
    F_tp = F[:, :, e_idx[0]] if e_idx.size else None   # time x pitch, linear units

    mask_fill   = ~np.isfinite(F_tp)
    mask_nonpos = np.isfinite(F_tp) & (F_tp <= 0)
    mask_ok     = F_tp > 0

    print("Time–Pitch @ 2.1 MeV:")
    print("  valid (>0):  {:.1f}%".format(mask_ok.mean()*100))
    print("  non-positive: {:.1f}%".format(mask_nonpos.mean()*100))
    print("  missing:      {:.1f}%".format(mask_fill.mean()*100))

    # Where are full-time gaps? (all pitch angles missing at that time)
    full_time_gap = (~mask_ok).all(axis=1)
    print("  hours with ALL pitch missing: ", full_time_gap.sum())


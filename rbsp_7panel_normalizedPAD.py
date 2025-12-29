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

# Energies for panels (MeV)
E_MAG_TL   = 1.079
E_REPT_TL2 = 2.1
E_REPT_TL3 = 4.2

# Use pitch bin nearest to this angle for Time–L / normalization:
PITCH_SELECT_DEG = 90.0

# Color limits (log10 flux per MeV) — will convert to keV by -3 for L–time plots
LOGR_MAG_TL   = (-4.0, 1.0)
LOGR_REPT_TL2 = ( 0.0, 4.0)
LOGR_REPT_TL3 = ( 0.0, 3.0)

# Render non-positive flux (≤0)?
SHOW_NONPOS_AS_FLOOR = True   # True: draw as lowest color; False: leave blank
EPS_FLOOR            = 1e-6   # tiny floor used when True (per MeV)

# X-axis tick density (hours between labeled ticks: 12, 6, 3, …)
TICK_STEP_HOURS = 12

# Layout
LEFT_MARGIN, RIGHT_MARGIN = 0.10, 0.89
TOP_MARGIN,  BOTTOM_MARGIN = 0.95, 0.08
HSPACE = 0.14
ENERGY_LABEL_X = -0.10

# Output filename (auto from dates)
OUT_NAME = f"rbsp_7panel_L_time_and_PAD_{START_DATE}_{END_DATE}_2.png"

# Fonts
FS_SUP   = 24
FS_LABEL = 20
FS_TICK  = 20
FS_CBAR  = 24

# ------------ OMNI numeric codes (1-based WORD → pass WORD-1) --------------
KP_CODE    = 38   # WORD 39: Kp*10
DST_CODE   = 40   # WORD 41: Dst (nT)
BZ_CODE    = 16   # WORD 17: Bz_GSM (nT)
PDYN_CODE  = 28   # WORD 29: Flow pressure P (nPa)
NP_CODE    = 24   # WORD 25: Proton density (cm^-3)
V_CODE     = 25   # WORD 26: Bulk speed (km/s)
# ==========================================================================

import re, sys
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import requests
import cdflib

# ------------------------------ small utils --------------------------------
def hourly_grid_utc(t0, t1):
    s = pd.Timestamp(t0)
    e = pd.Timestamp(t1)
    s = s.tz_localize("UTC") if s.tz is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tz is None else e.tz_convert("UTC")
    return pd.date_range(start=s.floor("h"), end=e.ceil("h"), freq="1h")

def to_day_offset(times_utc, base_midnight_utc):
    """Convert datetimes to fractional days since base (UTC)."""
    t = pd.to_datetime(times_utc, utc=True)
    base = pd.Timestamp(base_midnight_utc)
    base = base.tz_localize("UTC") if base.tz is None else base.tz_convert("UTC")
    return (t - base) / np.timedelta64(1, "D")

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

# ---------- helpers for energy interpolation & PAD normalization ------------
def _geom_interp_two(a0, a1, w):
    """Geometric interpolation (used for flux channels)."""
    m = np.isfinite(a0) & np.isfinite(a1) & (a0 > 0) & (a1 > 0)
    out = np.full_like(a0, np.nan, float)
    out[m] = np.exp(np.log(a0[m]) * (1 - w) + np.log(a1[m]) * w)
    left  = (a0 > 0) & ~np.isfinite(a1)
    right = (a1 > 0) & ~np.isfinite(a0)
    out[left]  = a0[left]
    out[right] = a1[right]
    return out

def nearest_pitch_idx(pitch_vec_deg, target_deg=90.0):
    pv = np.array(pitch_vec_deg, dtype=float).ravel()
    dist = np.minimum(np.abs(pv - target_deg), np.abs((pv % 360.0) - target_deg))
    return int(np.nanargmin(dist))

def flux_time_series_at_energy_pitch(F_tpe, E_vec, pitch_vec, Ereq_mev, pitch_deg=90.0):
    """Return linear flux time series at given energy (exact or log-geometric interp) and pitch angle."""
    E = np.asarray(E_vec, float)
    pidx = nearest_pitch_idx(pitch_vec, pitch_deg)
    idx = np.where(np.isclose(E, Ereq_mev, atol=1e-6))[0]
    if idx.size:
        return F_tpe[:, pidx, idx[0]].astype(float)
    if not (E.min() <= Ereq_mev <= E.max()):
        raise ValueError(f"Energy {Ereq_mev} MeV outside [{E.min()},{E.max()}] MeV")
    hi = int(np.searchsorted(E, Ereq_mev, side="right")); lo = hi - 1
    w  = (Ereq_mev - E[lo]) / (E[hi] - E[lo])
    f0 = F_tpe[:, pidx, lo].astype(float)
    f1 = F_tpe[:, pidx, hi].astype(float)
    return _geom_interp_two(f0, f1, w)

def flux_time_pitch_at_energy_linear(F_tpe, E_vec, Ereq_mev):
    """Return linear flux matrix (time x pitch) at given energy (exact or log-geometric interp)."""
    E = np.asarray(E_vec, float)
    idx = np.where(np.isclose(E, Ereq_mev, atol=1e-6))[0]
    if idx.size:
        return F_tpe[:, :, idx[0]].astype(float)
    if not (E.min() <= Ereq_mev <= E.max()):
        raise ValueError(f"Energy {Ereq_mev} MeV outside [{E.min()},{E.max()}] MeV")
    hi = int(np.searchsorted(E, Ereq_mev, side="right")); lo = hi - 1
    w  = (Ereq_mev - E[lo]) / (E[hi] - E[lo])
    f0 = F_tpe[:, :, lo].astype(float)
    f1 = F_tpe[:, :, hi].astype(float)
    M = np.empty_like(f0, float)
    for j in range(f0.shape[1]):  # for each pitch
        M[:, j] = _geom_interp_two(f0[:, j], f1[:, j], w)
    return M

def before_storm_mask(times_utc, storm_dt):
    if storm_dt is None:
        return np.ones(len(times_utc), dtype=bool)
    t = pd.to_datetime(times_utc, utc=True)
    return t < storm_dt

def safe_max(arr):
    v = np.nanmax(arr) if np.any(np.isfinite(arr)) else np.nan
    return v if (np.isfinite(v) and v > 0) else np.nan

# ----------------------------- CDF day loader -------------------------------
def load_day(path: Path):
    """
    Universal L3 pitch-angle loader (MagEIS or REPT).
    Returns dict:
      time (N,), L (N,), pitch (P,), energy_MeV (E,),
      flux_tpe_perMeV (N x P x E)
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

    # energy → MeV
    eunits = str(cdf.varattsget(energy_var).get("UNITS","")).lower()
    if "kev" in eunits:
        energy_MeV = E / 1000.0
        scale = "keV"
    elif "mev" in eunits or eunits == "":
        energy_MeV = E; scale = "MeV"
    elif "ev" in eunits:
        energy_MeV = E / 1e6; scale = "eV"
    else:
        energy_MeV = E / 1000.0 if E.max() > 200 else E; scale = "auto"

    # L or L_star
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

    # per MeV
    units = atts.get("UNITS", "cm^-2 s^-1 sr^-1 MeV^-1")
    if ("kev" in units.lower()) or (scale.lower()=="keV"):
        F_perMeV = F / 1000.0
    else:
        F_perMeV = F

    return dict(time=t, L=L, pitch=A, energy_MeV=np.asarray(energy_MeV, float),
                flux_tpe_perMeV=np.asarray(F_perMeV, float))

# ---------------------------- OMNIweb downloader ---------------------------
def omni_nx1_series(start_dt, end_dt, varcode):
    """
    Fetch a single OMNI variable via nx1.cgi (hourly).
      varcode: int code (preferred; e.g., 38, 40, 16, 28) or name string.
    Returns a pandas Series indexed by UTC hours (tz-aware).
    """
    base_urls = [
        "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi",
        "https://omniweb.sci.gsfc.nasa.gov/cgi/nx1.cgi",
    ]
    var_str = str(varcode)
    last_err = None
    for activity in ("RetrieveData", "ftp"):
        for base in base_urls:
            try:
                params = {
                    "activity": activity, "res": "hour", "spacecraft": "omni2",
                    "start_date": pd.Timestamp(start_dt).strftime("%Y%m%d"),
                    "end_date":   pd.Timestamp(end_dt).strftime("%Y%m%d"),
                    "maxdays": str((pd.Timestamp(end_dt)-pd.Timestamp(start_dt)).days + 1),
                    "vars": var_str, "view": 0, "nsum": 1, "paper": 0, "table": 0
                }
                r = requests.get(base, params=params, timeout=45)
                r.raise_for_status()
                # Try staging payload; else parse inline
                m = re.search(r'(https?://omniweb\.[\w\.]+/staging/[^"\']+\.(?:lst|txt))', r.text, re.I)
                if m:
                    text = requests.get(m.group(1), timeout=60).text
                else:
                    text = r.text  # inline

                times, vals = [], []
                for line in text.splitlines():
                    s = line.strip()
                    if not s or s[0] in "#;<":
                        continue
                    parts = s.split()
                    try:
                        # Format A: YYYY DOY HH val
                        if len(parts) >= 3 and parts[0].isdigit() and len(parts[1]) <= 3 and parts[2].isdigit():
                            yyyy, doy, hh = int(parts[0]), int(parts[1]), int(parts[2])
                            dt = datetime(yyyy, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy-1, hours=hh)
                            tail = parts[3:]
                        # Format B: YYYY MM DD HH val
                        elif len(parts) >= 4 and all(p.isdigit() for p in parts[:4]):
                            yyyy, mm, dd, hh = map(int, parts[:4])
                            dt = datetime(yyyy, mm, dd, hh, tzinfo=timezone.utc)
                            tail = parts[4:]
                        else:
                            continue
                        if not tail: continue
                        x = tail[-1]  # take last token as requested var
                        v = np.nan if x.upper() in ("NA","NAN") else float(x)
                        times.append(dt); vals.append(v)
                    except Exception:
                        continue

                if not times:
                    raise RuntimeError("OMNI parse produced no samples.")
                idx = pd.DatetimeIndex(times, tz="UTC").sort_values()
                ser = pd.Series(np.array(vals, float), index=idx).sort_index()
                return ser
            except Exception as e:
                last_err = e
                continue
    raise last_err if last_err else RuntimeError(f"OMNI request failed for var {var_str!r}")

def get_omni_series(start_dt, end_dt, candidates):
    """Try numeric code(s) then names; return first that works."""
    last_err = None
    for var in candidates:
        try:
            s = omni_nx1_series(start_dt, end_dt, var)
            if len(s): return s.sort_index()
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"OMNI: none of {candidates} worked.")

def pdyn_from_np_v(hours, start_dt, end_dt, alpha_frac=0.0):
    """Compute Pdyn [nPa] from Np [cm^-3] and V [km/s]; optional He via (1+4α)."""
    np_series = get_omni_series(start_dt, end_dt, [NP_CODE, "Np"])
    v_series  = get_omni_series(start_dt, end_dt, [V_CODE,  "V"])
    np_h = np_series.reindex(hours, method="ffill")
    v_h  = v_series .reindex(hours, method="ffill")
    pdyn = 1.6726e-6 * np_h * (v_h**2) * (1.0 + 4.0*alpha_frac)
    return pd.Series(pdyn.values, index=hours, name="Pdyn")

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
    mag_L     = np.concatenate([x["L"]    for x in mag_parts])
    mag_E     = mag_parts[0]["energy_MeV"]
    mag_A     = mag_parts[0]["pitch"]
    mag_F     = np.concatenate([x["flux_tpe_perMeV"] for x in mag_parts], axis=0)

    # ---------- REPT ----------
    rept_paths = []
    for d in daterange(start, end):
        ymd = d.strftime("%Y%m%d")
        p = find_daily_file(REPT_BASE, ymd, PROBE, REPT_TAG)
        if p and p.exists(): rept_paths.append(p)
    if not rept_paths: sys.exit("[ABORT] No REPT files found.")
    rept_parts = [load_day(p) for p in rept_paths]
    rept_time  = np.concatenate([x["time"] for x in rept_parts])
    rept_L     = np.concatenate([x["L"]    for x in rept_parts])
    rept_E     = rept_parts[0]["energy_MeV"]
    rept_A     = rept_parts[0]["pitch"]
    rept_F     = np.concatenate([x["flux_tpe_perMeV"] for x in rept_parts], axis=0)
    # (If you want the original REPT scaling by 0.5, uncomment:)
    rept_F *= 0.5

    # ---------- series/maps ----------
    # L–time (log10 per MeV)
    mag_log_tl1, mag_how = to_log10_with_floor(
        flux_time_series_at_energy_pitch(mag_F,  mag_E, mag_A,  E_MAG_TL,  pitch_deg=PITCH_SELECT_DEG),
        show_floor=SHOW_NONPOS_AS_FLOOR, eps=EPS_FLOOR
    ), "series"
    rept_log_tl2, h2 = to_log10_with_floor(
        flux_time_series_at_energy_pitch(rept_F, rept_E, rept_A, E_REPT_TL2, pitch_deg=PITCH_SELECT_DEG),
        show_floor=SHOW_NONPOS_AS_FLOOR, eps=EPS_FLOOR
    ), "series"
    rept_log_tl3, h3 = to_log10_with_floor(
        flux_time_series_at_energy_pitch(rept_F, rept_E, rept_A, E_REPT_TL3, pitch_deg=PITCH_SELECT_DEG),
        show_floor=SHOW_NONPOS_AS_FLOOR, eps=EPS_FLOOR
    ), "series"

    # PAD (linear) at energies
    mag_PAD_lin   = flux_time_pitch_at_energy_linear(mag_F,  mag_E, E_MAG_TL)     # (Nt, Npitch)
    rept_PAD_lin2 = flux_time_pitch_at_energy_linear(rept_F, rept_E, E_REPT_TL2)  # (Nt, Npitch)
    rept_PAD_lin3 = flux_time_pitch_at_energy_linear(rept_F, rept_E, E_REPT_TL3)  # (Nt, Npitch)

    # Time bounds (tz-aware)
    t_mag  = pd.to_datetime(mag_time,  utc=True)
    t_rept = pd.to_datetime(rept_time, utc=True)
    xmin = min(t_mag.min(), t_rept.min())
    xmax = max(t_mag.max(), t_rept.max())

    # ---------- x-axis as "days since start" ----------
    base_day_utc = xmin.floor("D")  # tz-aware UTC
    x_mag   = to_day_offset(t_mag,  base_day_utc)
    x_rept  = to_day_offset(t_rept, base_day_utc)
    hours   = hourly_grid_utc(xmin, xmax)
    x_hours = to_day_offset(hours, base_day_utc)

    # Labeled ticks every TICK_STEP_HOURS, plus minor grid at half-step
    day_span = (xmax.ceil("D") - base_day_utc) / pd.Timedelta(days=1)
    n_days   = int(np.ceil(float(day_span)))
    max_day  = float(n_days)

    tick_times = pd.date_range(
        start=base_day_utc,
        end=base_day_utc + pd.Timedelta(days=n_days),
        freq=f"{TICK_STEP_HOURS}h"
    )
    tick_pos = to_day_offset(tick_times, base_day_utc)
    tick_lab = tick_times.strftime("%b %d\n%H:%M").tolist()

    minor_step_days = (TICK_STEP_HOURS / 24.0) / 2.0
    minor_loc = mticker.MultipleLocator(minor_step_days)

    # ---------- OMNI: Kp/Dst & storm time ----------
    kp_dst = None
    try:
        kp_raw  = get_omni_series(hours[0], hours[-1], [KP_CODE]) / 10.0  # → Kp
        dst_raw = get_omni_series(hours[0], hours[-1], [DST_CODE])
        kp  = kp_raw .reindex(hours, method="ffill").rename("Kp")
        dst = dst_raw.reindex(hours, method="ffill").rename("Dst")
        kp_dst = pd.DataFrame({"Kp": kp, "Dst": dst}).sort_index()
        print(f"[ok] Kp/Dst: {len(kp_dst)} pts")
    except Exception as e:
        print(f"[warn] Kp/Dst failed: {e}")

    # --- storm time detection (first time Dst <= -30) ---
    storm_time = None
    if kp_dst is not None:
        mstorm = (kp_dst["Dst"] <= -30).fillna(False)   # boolean Series
        if mstorm.any():
            # take the first timestamp where condition is True (label-based, safe)
            storm_time = mstorm[mstorm].index[0]
            print(f"[storm] First Dst<=-30 at {storm_time}")
        else:
            print("[storm] No Dst<=-30 found in interval.")

    # ---------- Pre-storm normalization for PADs (use 90° series) ----------
    mag_f90     = flux_time_series_at_energy_pitch(mag_F,  mag_E, mag_A,  E_MAG_TL,  pitch_deg=90.0)
    rept_f90_2  = flux_time_series_at_energy_pitch(rept_F, rept_E, rept_A, E_REPT_TL2, pitch_deg=90.0)
    rept_f90_3  = flux_time_series_at_energy_pitch(rept_F, rept_E, rept_A, E_REPT_TL3, pitch_deg=90.0)

    m_mag_pre   = before_storm_mask(mag_time,  storm_time)
    m_rept_pre  = before_storm_mask(rept_time, storm_time)

    norm_mag = safe_max(mag_f90[m_mag_pre])
    norm_r2  = safe_max(rept_f90_2[m_rept_pre])
    norm_r3  = safe_max(rept_f90_3[m_rept_pre])

    mag_PAD_norm   = mag_PAD_lin   / norm_mag if np.isfinite(norm_mag) else np.full_like(mag_PAD_lin,   np.nan)
    rept_PAD_norm2 = rept_PAD_lin2 / norm_r2   if np.isfinite(norm_r2)  else np.full_like(rept_PAD_lin2, np.nan)
    rept_PAD_norm3 = rept_PAD_lin3 / norm_r3   if np.isfinite(norm_r3)  else np.full_like(rept_PAD_lin3, np.nan)

    # ------------------------------ PLOT (7 panels) --------------------------
    # L vs time panels will display per-keV (convert from log10 per-MeV by subtracting 3)
    LOGR_MAG_TL_keV   = (LOGR_MAG_TL[0] - 3.0,   LOGR_MAG_TL[1] - 3.0)
    LOGR_REPT_TL2_keV = (LOGR_REPT_TL2[0] - 3.0, LOGR_REPT_TL2[1] - 3.0)
    LOGR_REPT_TL3_keV = (LOGR_REPT_TL3[0] - 3.0, LOGR_REPT_TL3[1] - 3.0)

    mag_log_tl1_keV  = mag_log_tl1  - 3.0
    rept_log_tl2_keV = rept_log_tl2 - 3.0
    rept_log_tl3_keV = rept_log_tl3 - 3.0

    fig, axes = plt.subplots(
        7, 1, figsize=(12, 18), sharex=True,
        gridspec_kw={"height_ratios":[1,1,1, 1,1,1, 0.8]}
    )
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        top=TOP_MARGIN, bottom=BOTTOM_MARGIN, hspace=HSPACE)
    fig.suptitle(
        f"{PROBE.upper()} L vs Time (per keV) & PAD (normalized to pre-storm 90° max)",
        fontsize=FS_SUP
    )

    extend_kw = 'min' if SHOW_NONPOS_AS_FLOOR else 'neither'
    CBAR_LABEL_KEV  = r"log$_{10}$(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)"
    CBAR_LABEL_NORM = "Normalized flux (× pre-storm 90° max)"

    # Panel 1: L vs Time — MagEIS 1.079 MeV (per keV)
    ax = axes[0]
    ok = np.isfinite(mag_log_tl1_keV) & np.isfinite(mag_L)
    sc1 = ax.scatter(x_mag[ok], mag_L[ok], c=mag_log_tl1_keV[ok], s=6, cmap="jet",
                     vmin=LOGR_MAG_TL_keV[0], vmax=LOGR_MAG_TL_keV[1])
    ax.set_ylabel("L")
    if np.any(ok):
        lo, hi = np.nanpercentile(mag_L[ok], [1,99]); lo=max(0.8,lo); hi=min(10.0,hi+0.15*(hi-lo))
        ax.set_ylim(lo, hi)
    ax.set_xlim(0.0, max_day)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_MAG_TL:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    fig.colorbar(sc1, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw).ax.tick_params(labelsize=FS_TICK)

    # Panel 2: L vs Time — REPT 2.1 MeV (per keV)
    ax = axes[1]
    ok = np.isfinite(rept_log_tl2_keV) & np.isfinite(rept_L)
    sc2 = ax.scatter(x_rept[ok], rept_L[ok], c=rept_log_tl2_keV[ok], s=6, cmap="jet",
                     vmin=LOGR_REPT_TL2_keV[0], vmax=LOGR_REPT_TL2_keV[1])
    ax.set_ylabel("L"); ax.set_xlim(0.0, max_day)
    if np.any(ok):
        lo, hi = np.nanpercentile(rept_L[ok], [1,99]); lo=max(0.8,lo); hi=min(10.0,hi+0.15*(hi-lo))
        ax.set_ylim(lo, hi)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL2:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    fig.colorbar(sc2, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw).ax.tick_params(labelsize=FS_TICK)

    # Panel 3: L vs Time — REPT 4.2 MeV (per keV)
    ax = axes[2]
    ok = np.isfinite(rept_log_tl3_keV) & np.isfinite(rept_L)
    sc3 = ax.scatter(x_rept[ok], rept_L[ok], c=rept_log_tl3_keV[ok], s=6, cmap="jet",
                     vmin=LOGR_REPT_TL3_keV[0], vmax=LOGR_REPT_TL3_keV[1])
    ax.set_ylabel("L"); ax.set_xlim(0.0, max_day)
    if np.any(ok):
        lo, hi = np.nanpercentile(rept_L[ok], [1,99]); lo=max(0.8,lo); hi=min(10.0,hi+0.15*(hi-lo))
        ax.set_ylim(lo, hi)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL3:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    fig.colorbar(sc3, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw).ax.tick_params(labelsize=FS_TICK)

    # Shared colorbar label for panels 1–3
    fig.text(0.93, 0.675, CBAR_LABEL_KEV, rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # before plotting panels 4–6
    def vmax_linear(M):
        # 99th percentile, minimum of 1.0, cap at 3.0 to avoid blowouts
        q = np.nanpercentile(M, 99)
        return float(max(1.0, min(q, 3.0)))

    vmax4 = vmax_linear(mag_PAD_norm)
    vmax5 = vmax_linear(rept_PAD_norm2)
    vmax6 = vmax_linear(rept_PAD_norm3)

    # Panel 4: PAD — MagEIS 1.079 MeV (normalized, log10)
    ax = axes[3]
    pm4 = ax.pcolormesh(x_mag, mag_A, mag_PAD_norm.T, shading="nearest", cmap="jet",
                    vmin=0.2, vmax=1)   # 0..1 relative to pre-storm 90° max
    ax.set_ylabel("Pitch (deg)"); ax.set_xlim(0.0, max_day)
    fig.colorbar(pm4, ax=ax, pad=0.01, fraction=0.04).ax.tick_params(labelsize=FS_TICK)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_MAG_TL:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # Panel 5: PAD — REPT 2.1 MeV (normalized, log10)
    ax = axes[4]
    pm5 = ax.pcolormesh(x_rept, rept_A, rept_PAD_norm2.T, shading="nearest", cmap="jet",
                    vmin=0.2, vmax=1)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlim(0.0, max_day)
    fig.colorbar(pm5, ax=ax, pad=0.01, fraction=0.04).ax.tick_params(labelsize=FS_TICK)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL2:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # Panel 6: PAD — REPT 4.2 MeV (normalized, log10)
    ax = axes[5]
    pm6 = ax.pcolormesh(x_rept, rept_A, rept_PAD_norm3.T, shading="nearest", cmap="jet",
                    vmin=0.2, vmax=1)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlim(0.0, max_day)
    fig.colorbar(pm6, ax=ax, pad=0.01, fraction=0.04).ax.tick_params(labelsize=FS_TICK)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL3:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)

    # Shared colorbar label for panels 4–6
    fig.text(0.93, 0.33, CBAR_LABEL_NORM, rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # Panel 7: Kp & Dst
    ax7 = axes[6]
    ax7.grid(True, alpha=0.3, linestyle=":")
    ax7.set_ylabel("Dst (nT)")
    if kp_dst is not None:
        df = kp_dst.copy().reindex(hours).interpolate(limit_direction="both")
        ax7.plot(x_hours, df["Dst"].values, color="k", lw=1.5, label="Dst")
        ax7r = ax7.twinx()
        ax7r.step(x_hours, df["Kp"].values, where="post", color="tab:orange", lw=1.5, label="Kp")
        ax7r.set_ylabel("Kp", color="tab:orange")
        ax7r.tick_params(axis='y', labelcolor="tab:orange")
        ax7r.set_ylim(0, 9)
        if storm_time is not None:
            xs = float(to_day_offset([storm_time], base_day_utc)[0])
            ax7.axvline(xs, color="crimson", ls="--", lw=1, alpha=0.8)
            ax7.text(xs, 0.05, "Dst≤-30", transform=ax7.get_xaxis_transform(),
                     ha="left", va="bottom", fontsize=FS_LABEL*0.8, color="crimson")
        lines, labels = [], []
        for axx in (ax7, ax7r):
            h, l = axx.get_legend_handles_labels()
            lines += h; labels += l
        ax7.legend(lines, labels, loc="upper right")
    else:
        ax7.text(0.5,0.5,"No Kp/Dst", transform=ax7.transAxes, ha="center", va="center")

    # X-axis ticks, labels, and minor grid on ALL panels
    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.xaxis.set_minor_locator(minor_loc)
        ax.grid(which="minor", alpha=0.15, linestyle=":")
        ax.set_xlim(0.0, max_day)
        ax.margins(x=0)
    axes[-1].set_xlabel("Day (UTC)")

    fig.savefig(OUT_NAME, dpi=180, bbox_inches="tight", pad_inches=0.02)
    print(f"[SAVED] {OUT_NAME}")
    plt.show()

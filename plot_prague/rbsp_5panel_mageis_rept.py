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
E_MAG_TL   = 0.909
E_REPT_TL2 = 2.1
E_REPT_TL3 = 4.2
# (No PAD panel anymore, E_REPT_TP removed)

# Use pitch bin nearest to this angle for Time–L:
PITCH_SELECT_DEG = 90.0

# Color limits (log10 flux per keV) -------------- NOTE: ranges are unchanged
LOGR_MAG_TL   = (0, 4)
LOGR_REPT_TL2 = (-1, 3)
LOGR_REPT_TL3 = (-2, 2)

# Render non-positive flux (≤0)?
SHOW_NONPOS_AS_FLOOR = True   # True: draw as lowest color; False: leave blank
EPS_FLOOR            = 1e-6   # tiny floor used when True (per keV)

# X-axis tick density (hours between labeled ticks: 12, 6, 3, …)
TICK_STEP_HOURS = 12

# Layout
LEFT_MARGIN, RIGHT_MARGIN = 0.10, 0.89
TOP_MARGIN,  BOTTOM_MARGIN = 0.95, 0.08
HSPACE = 0.14
ENERGY_LABEL_X = -0.10

# Output filename (auto from dates)
OUT_NAME = f"/home/zhouyu/read_cdf/plot_prague/result/rbsp_5panel_mageis_rept_{START_DATE}_{END_DATE}_perkeV_2.png"

# Fonts
FS_SUP   = 28
FS_LABEL = 24
FS_TICK  = 24
FS_CBAR  = 28

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

# ----------------------------- CDF day loader -------------------------------
def load_day(path: Path):
    """
    Universal L3 pitch-angle loader (MagEIS or REPT).
    Returns dict:
      time (N,), L (N,), pitch (P,), energy_MeV (E,),
      flux_tpe_perkeV (N x P x E)     # <-- unified to per keV
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

    # energy → MeV (for the E axis only)
    eunits = str(cdf.varattsget(energy_var).get("UNITS","")).lower()
    if "kev" in eunits:
        energy_MeV = E / 1000.0; scale = "keV"
    elif "mev" in eunits or eunits == "":
        energy_MeV = E;           scale = "mev"
    elif "ev" in eunits:
        energy_MeV = E / 1e6;     scale = "ev"
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

    # ---- convert flux units → per keV (careful and explicit) ---------------
    units_raw = atts.get("UNITS", "") or ""
    u = units_raw.lower()
    if "kev" in u:
        F_perkeV = F
    elif "mev" in u:
        F_perkeV = F / 1000.0          # per MeV → per keV
    elif re.search(r"\bev\b", u):
        F_perkeV = F * 1000.0          # per eV → per keV
    else:
        # Fallback: infer from energy scale if units are blank/odd
        if scale == "keV":
            F_perkeV = F
        elif scale == "mev":
            F_perkeV = F / 1000.0
        elif scale == "ev":
            F_perkeV = F * 1000.0
        else:
            F_perkeV = F  # leave as-is if uncertain

    return dict(time=t, L=L, pitch=A, energy_MeV=np.asarray(energy_MeV, float),
                flux_tpe_perkeV=np.asarray(F_perkeV, float))

# ------------------------ extraction & interpolation ------------------------
def nearest_pitch_idx(pitch_vec_deg, target_deg=90.0):
    pv = np.array(pitch_vec_deg, dtype=float).ravel()
    dist = np.minimum(np.abs(pv - target_deg), np.abs((pv % 360.0) - target_deg))
    return int(np.nanargmin(dist))

def timeL_at_energy(F_tpe, E_vec, pitch_vec, Ereq_mev, pitch_deg=90.0, exact_only=False):
    E = np.asarray(E_vec, dtype=float)
    pidx = nearest_pitch_idx(pitch_vec, pitch_deg)
    idx = np.where(np.isclose(E, Ereq_mev, atol=1e-6))[0]
    if idx.size:
        f = F_tpe[:, pidx, idx[0]].astype(float); how = "exact"
    else:
        if not (E.min() <= Ereq_mev <= E.max()):
            raise ValueError(f"Energy {Ereq_mev} MeV outside [{E.min()},{E.max()}] MeV")
        if exact_only:
            raise ValueError(f"Energy {Ereq_mev} MeV not an exact channel.")
        hi = int(np.searchsorted(E, Ereq_mev, side="right")); lo = hi - 1
        w  = (Ereq_mev - E[lo]) / (E[hi] - E[lo])
        f0 = F_tpe[:, pidx, lo].astype(float); f1 = F_tpe[:, pidx, hi].astype(float)
        f  = np.full_like(f0, np.nan, float)
        m  = (f0 > 0) & (f1 > 0)
        f[m] = np.exp(np.log(f0[m]) * (1-w) + np.log(f1[m]) * w)
        left  = (f0 > 0) & ~np.isfinite(f1); right = (f1 > 0) & ~np.isfinite(f0)
        f[left]  = f0[left]; f[right] = f1[right]
        how = f"interp {E[lo]:g}-{E[hi]:g}"
    logf = to_log10_with_floor(f, show_floor=SHOW_NONPOS_AS_FLOOR, eps=EPS_FLOOR)
    return logf, how

def time_pitch_at_energy(F_tpe, E_vec, Ereq_mev):
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
        "font.weight": "bold",
        "axes.labelweight": "bold",
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
    rept_L     = np.concatenate([x["L"]    for x in rept_parts])
    rept_E     = rept_parts[0]["energy_MeV"]
    rept_A     = rept_parts[0]["pitch"]
    rept_F     = np.concatenate([x["flux_tpe_perkeV"] for x in rept_parts], axis=0)  # per keV
    # --- show REPT as flux/2 ---
    rept_F *= 0.5

    # ---------- series/maps ----------
    mag_log_tl1, mag_how = timeL_at_energy(mag_F, mag_E, mag_A, E_MAG_TL, pitch_deg=PITCH_SELECT_DEG)
    print(f"[MagEIS] Time–L @ {E_MAG_TL} MeV → {mag_how}")

    rept_log_tl2, h2 = timeL_at_energy(rept_F, rept_E, rept_A, E_REPT_TL2, pitch_deg=PITCH_SELECT_DEG)
    rept_log_tl3, h3 = timeL_at_energy(rept_F, rept_E, rept_A, E_REPT_TL3, pitch_deg=PITCH_SELECT_DEG)
    print(f"[REPT]   Time–L @ {E_REPT_TL2} MeV → {h2}")
    print(f"[REPT]   Time–L @ {E_REPT_TL3} MeV → {h3}")

    # (PAD panel removed)

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
        start=base_day_utc,  # already tz-aware; do NOT tz_localize
        end=base_day_utc + pd.Timedelta(days=n_days),
        freq=f"{TICK_STEP_HOURS}h"
    )
    tick_pos = to_day_offset(tick_times, base_day_utc)
    tick_lab = tick_times.strftime("%b %d\n%H:%M").tolist()

    minor_step_days = (TICK_STEP_HOURS / 24.0) / 2.0
    minor_loc = mticker.MultipleLocator(minor_step_days)

    # ---------- OMNI: Kp/Dst & Bz/Pdyn ----------
    kp_dst = None
    try:
        kp_raw  = get_omni_series(hours[0], hours[-1], [KP_CODE]) / 10.0  # → Kp
        dst_raw = get_omni_series(hours[0], hours[-1], [DST_CODE])
        kp  = kp_raw .reindex(hours, method="ffill").rename("Kp")
        dst = dst_raw.reindex(hours, method="ffill").rename("Dst")
        kp_dst = pd.DataFrame({"Kp": kp, "Dst": dst})
        print(f"[ok] Kp/Dst: {len(kp_dst)} pts")
    except Exception as e:
        print(f"[warn] Kp/Dst failed: {e}")

    pdyn_bz = None
    try:
        bz_raw = get_omni_series(hours[0], hours[-1], [BZ_CODE])
        bz = bz_raw.reindex(hours, method="ffill").rename("Bz")
        try:
            pdyn_raw = get_omni_series(hours[0], hours[-1], [PDYN_CODE])
            pdyn = pdyn_raw.reindex(hours, method="ffill").rename("Pdyn")
        except Exception:
            print("[info] OMNI code for P failed; computing Pdyn from Np & V")
            pdyn = pdyn_from_np_v(hours, hours[0], hours[-1], alpha_frac=0.0)
        pdyn_bz = pd.concat([bz, pdyn], axis=1)
        print(f"[ok] Bz/Pdyn: {len(pdyn_bz)} pts")
    except Exception as e:
        print(f"[warn] Bz/Pdyn failed: {e}")

    # ------------------------------ PLOT -------------------------------------
    fig, axes = plt.subplots(
        5, 1, figsize=(12, 12), sharex=True,   # ← 5 panels (4th removed)
        gridspec_kw={"height_ratios":[1,1,1,0.9,0.9]}
    )
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        top=TOP_MARGIN, bottom=BOTTOM_MARGIN, hspace=HSPACE)
    fig.suptitle(
        f"{PROBE.upper()} Electron Flux Observation (Pitch={PITCH_SELECT_DEG:.0f}°)",
        fontsize=FS_SUP,fontweight="bold"
    )

    extend_kw = 'min' if SHOW_NONPOS_AS_FLOOR else 'neither'
    CBAR_LABEL = r"log$_{10}$(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)"  # ← per keV

    # Panel 1: MagEIS 0.909 MeV
    ax = axes[0]
    ok = np.isfinite(mag_log_tl1) & np.isfinite(mag_L)
    sc1 = ax.scatter(x_mag[ok], mag_L[ok], c=mag_log_tl1[ok], s=6, cmap="jet",
                     vmin=LOGR_MAG_TL[0], vmax=LOGR_MAG_TL[1])
    ax.set_ylabel("L")
    if np.any(ok):
        lo, hi = np.nanpercentile(mag_L[ok], [1,99]); lo=max(0.8,lo); hi=min(10.0,hi+0.15*(hi-lo))
        ax.set_ylim(lo, hi)
    ax.set_xlim(0.0, max_day)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_MAG_TL:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    cb = fig.colorbar(sc1, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw); cb.ax.tick_params(labelsize=FS_TICK)

    # Panel 2: REPT 2.1 MeV
    ax = axes[1]
    ok = np.isfinite(rept_log_tl2) & np.isfinite(rept_L)
    sc2 = ax.scatter(x_rept[ok], rept_L[ok], c=rept_log_tl2[ok], s=6, cmap="jet",
                     vmin=LOGR_REPT_TL2[0], vmax=LOGR_REPT_TL2[1])
    ax.set_ylabel("L"); ax.set_xlim(0.0, max_day)
    if np.any(ok):
        lo, hi = np.nanpercentile(rept_L[ok], [1,99]); lo=max(0.8,lo); hi=min(10.0,hi+0.15*(hi-lo))
        ax.set_ylim(lo, hi)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL2:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    cb = fig.colorbar(sc2, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw); cb.ax.tick_params(labelsize=FS_TICK)

    # Panel 3: REPT 4.2 MeV
    ax = axes[2]
    ok = np.isfinite(rept_log_tl3) & np.isfinite(rept_L)
    sc3 = ax.scatter(x_rept[ok], rept_L[ok], c=rept_log_tl3[ok], s=6, cmap="jet",
                     vmin=LOGR_REPT_TL3[0], vmax=LOGR_REPT_TL3[1])
    ax.set_ylabel("L"); ax.set_xlim(0.0, max_day)
    if np.any(ok):
        lo, hi = np.nanpercentile(rept_L[ok], [1,99]); lo=max(0.8,lo); hi=min(10.0,hi+0.15*(hi-lo))
        ax.set_ylim(lo, hi)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL3:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    cb = fig.colorbar(sc3, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw); cb.ax.tick_params(labelsize=FS_TICK)

    # Shared cb label for the three above:
    fig.text(0.93, 0.68, CBAR_LABEL, rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # Panel 4: Dst & Kp (x_hours)
    ax5 = axes[3]
    ax5.grid(True, alpha=0.3, linestyle=":")
    ax5.set_ylabel("Dst (nT)")
    if kp_dst is not None:
        df = kp_dst.copy().reindex(hours).interpolate(limit_direction="both")
        ax5.plot(x_hours, df["Dst"].values, color="k", lw=1.5, label="Dst")
        ax5r = ax5.twinx()
        ax5r.step(x_hours, df["Kp"].values, where="post", color="tab:orange", lw=1.5, label="Kp")
        ax5r.set_ylabel("Kp", color="tab:orange")
        ax5r.tick_params(axis='y', labelcolor="tab:orange")
        ax5r.set_ylim(0, 9)
        lines, labels = [], []
        for axx in (ax5, ax5r):
            h, l = axx.get_legend_handles_labels()
            lines += h; labels += l
        ax5.legend(lines, labels, loc="upper right")
    else:
        ax5.text(0.5,0.5,"No Kp/Dst", transform=ax5.transAxes, ha="center", va="center")

    # Panel 5: Pdyn & Bz (x_hours)
    ax6 = axes[4]
    ax6.grid(True, alpha=0.3, linestyle=":")
    ax6.set_ylabel("Pdyn (nPa)")
    if pdyn_bz is not None:
        df = pdyn_bz.copy().reindex(hours).interpolate(limit_direction="both")
        ax6.plot(x_hours, df["Pdyn"].values, color="tab:blue", lw=1.5, label="Pdyn")
        ax6r = ax6.twinx()
        ax6r.plot(x_hours, df["Bz"].values, color="tab:red", lw=1.2, label="Bz")
        ax6r.set_ylabel("Bz (nT)", color="tab:red")
        ax6r.tick_params(axis='y', labelcolor="tab:red")
        lines, labels = [], []
        for axx in (ax6, ax6r):
            h, l = axx.get_legend_handles_labels()
            lines += h; labels += l
        ax6.legend(lines, labels, loc="upper right")
    else:
        ax6.text(0.5,0.5,"No Pdyn/Bz", transform=ax6.transAxes, ha="center", va="center")

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

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

# Use pitch bin nearest to this angle for Time–L:
PITCH_SELECT_DEG = 70.0

# Color limits (log10 flux per keV) for Time–L
LOGR_MAG_TL   = (-4.0, 1.0)
LOGR_REPT_TL2 = (0, 4)
LOGR_REPT_TL3 = (0, 3)

# -------- Normalized PAD options (Panels 4–6) ------------------------------
NORM_PAD_LOG   = False          # False → color = F/F_90 (linear). True → log10(F/F_90)
LOGR_LP_NORM   = (-2.0, 2.0)    # used when NORM_PAD_LOG=True
LINR_LP_NORM   = (0.2, 5.0)     # used when NORM_PAD_LOG=False
REF_SEARCH_HALFSPAN = 45.0      # search ±deg around 90° for a valid reference
PITCH_INTERP_ENABLE = True      # fill small NaN gaps along pitch BEFORE normalizing
PITCH_INTERP_MAX_GAP_DEG = 12.0 # only fill gaps narrower than this many degrees

# L-bin settings for PAD pcolormesh
L_MIN, L_MAX, L_BIN_WIDTH = 0.8, 10.0, 0.1

# Render non-positive flux (≤0)?
SHOW_NONPOS_AS_FLOOR = True
EPS_FLOOR            = 1e-6

# X-axis tick density (hours between labeled ticks: 12, 6, 3, …)
TICK_STEP_HOURS = 12

# Layout
LEFT_MARGIN, RIGHT_MARGIN = 0.10, 0.89
TOP_MARGIN,  BOTTOM_MARGIN = 0.95, 0.08
HSPACE = 0.14
ENERGY_LABEL_X = -0.10

# Output filename (auto from dates)
OUT_NAME = f"rbsp_7panel_mageis_rept_{START_DATE}_{END_DATE}_perkeV_normPA_pmesh.png"

# Fonts
FS_SUP   = 24
FS_LABEL = 20
FS_TICK  = 20
FS_CBAR  = 24

# ------------ OMNI numeric codes (1-based WORD → pass WORD-1) --------------
KP_CODE    = 38   # WORD 39: Kp*10
DST_CODE   = 40   # WORD 41: Dst (nT)
BZ_CODE    = 16   # not used here
PDYN_CODE  = 28   # not used here
NP_CODE    = 24   # not used here
V_CODE     = 25   # not used here
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
    s = pd.Timestamp(t0); e = pd.Timestamp(t1)
    s = s.tz_localize("UTC") if s.tz is None else s.tz_convert("UTC")
    e = e.tz_localize("UTC") if e.tz is None else e.tz_convert("UTC")
    return pd.date_range(start=s.floor("h"), end=e.ceil("h"), freq="1h")

def to_day_offset(times_utc, base_midnight_utc):
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

# ---------------- pitch helpers for pcolormesh ------------------------------
def pitch_edges_from_centers(pitch_centers):
    c = np.asarray(pitch_centers, float)
    dc = np.diff(c)
    left  = np.concatenate(([c[0] - dc[0]/2], c[:-1] + dc/2))
    right = np.concatenate((c[:-1] + dc/2, [c[-1] + dc[-1]/2]))
    edges = np.clip(np.concatenate(([left[0]], c[:-1] + dc/2, [right[-1]])), 0, 180)
    # fix monotonicity
    edges = np.r_[max(0.0, c[0] - abs(dc[0])/2), (c[:-1] + c[1:])/2, min(180.0, c[-1] + abs(dc[-1])/2)]
    return edges

# ----------------------------- CDF day loader -------------------------------
def load_day(path: Path):
    """
    Universal L3 pitch-angle loader (MagEIS or REPT).
    Returns dict:
      time (N,), L (N,), pitch (P,), energy_MeV (E,),
      flux_tpe_perkeV (N x P x E)
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

    # ---- force flux to per keV ---------------------------------------------
    units_raw = atts.get("UNITS", "cm^-2 s^-1 sr^-1 keV^-1")
    units = (units_raw or "").lower()
    if "mev" in units:
        F_perkeV = F / 1000.0
    elif "kev" in units:
        F_perkeV = F
    elif " ev" in units or "eV" in units_raw:
        F_perkeV = F / 1e3
    else:
        F_perkeV = F if scale == "keV" else (F / 1000.0 if scale == "mev" else F)

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

def time_pitch_linear_at_energy(F_tpe, E_vec, Ereq_mev):
    """Return time×pitch matrix at requested energy (linear, per keV)."""
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
        M  = np.full_like(f0, np.nan, float)
        m  = (f0 > 0) & (f1 > 0)
        M[m] = np.exp(np.log(f0[m]) * (1-w) + np.log(f1[m]) * w)
        left  = (f0 > 0) & ~np.isfinite(f1); right = (f1 > 0) & ~np.isfinite(f0)
        M[left]  = f0[left]; M[right] = f1[right]
    return M

# ---------------- Pitch-gap filling (row-wise, along pitch) -----------------
def _interp_pitch_row(pitch_deg, row_vals, max_gap_deg=12.0):
    """
    Interpolate NaNs along pitch for a single time-row.
    Fill interior gaps whose span ≤ max_gap_deg; keep large gaps & edges NaN.
    """
    p = np.asarray(pitch_deg, float)
    y = np.asarray(row_vals, float).copy()
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2:
        return y
    idx = np.where(ok)[0]
    y_lin = np.interp(p, p[idx], y[idx])
    # mask large gaps and edges
    nan_mask = ~ok
    if np.any(nan_mask):
        gaps = []
        start = None
        for i, bad in enumerate(nan_mask):
            if bad and start is None:
                start = i
            if (not bad or i == len(nan_mask)-1) and start is not None:
                end = i if not bad else i+1
                gaps.append((start, end))
                start = None
        for a, b in gaps:
            span = abs(p[min(b, len(p)-1)] - p[max(a-1,0)])
            if span > max_gap_deg or a == 0 or b == len(p):
                y_lin[a:b] = np.nan
    return y_lin

def _fill_pitch_gaps(M, pitch_deg, max_gap_deg=12.0):
    if not PITCH_INTERP_ENABLE:
        return M
    T, _ = M.shape
    out = np.empty_like(M)
    for i in range(T):
        out[i, :] = _interp_pitch_row(pitch_deg, M[i, :], max_gap_deg=max_gap_deg)
    return out

# ---------------- Build PAD grid (L×Pitch) for pcolormesh -------------------
def build_pad_grid(F_tpe, E_vec, pitch_vec, L_vec, Ereq_mev,
                   target_deg=90.0, search_halfspan=45.0, max_gap_deg=12.0,
                   l_min=0.8, l_max=10.0, l_bin_width=0.1,
                   take_log=False):
    """
    Returns:
      L_edges      (Nx+1,)
      pitch_edges  (Ny+1,)
      Z_color      (Ny, Nx)   # either ratio or log10(ratio) ready for pcolormesh
    """
    # 1) time×pitch at energy (linear per keV), fill small pitch gaps
    M_raw = time_pitch_linear_at_energy(F_tpe, E_vec, Ereq_mev)    # (T,P)
    M = _fill_pitch_gaps(M_raw, pitch_vec, max_gap_deg=max_gap_deg)
    T, P = M.shape
    pitch = np.asarray(pitch_vec, float)

    # 2) choose reference (per row) near 90°, within window
    pidx90 = nearest_pitch_idx(pitch, target_deg)
    ref_idx = np.full(T, -1, dtype=int)
    for i in range(T):
        ok = np.isfinite(M[i, :]) & (M[i, :] > 0)
        if ok.any():
            cand = ok & (np.abs(pitch - target_deg) <= search_halfspan)
            if cand.any():
                idxs = np.where(cand)[0]
                ref_idx[i] = int(idxs[np.argmin(np.abs(pitch[idxs] - target_deg))])
    ref = np.full(T, np.nan, float)
    good = ref_idx >= 0
    ref[good] = M[np.arange(T)[good], ref_idx[good]]

    # 3) ratio (broadcast) and mask invalids
    with np.errstate(invalid="ignore", divide="ignore"):
        R = M / ref[:, None]
        bad = (~np.isfinite(M)) | (M <= 0) | (~np.isfinite(ref)[:, None]) | (ref[:, None] <= 0)
        R[bad] = np.nan
    # cosmetic: exact 1 at chosen ref bin
    rows = np.arange(T)[good]; cols = ref_idx[good]
    num_ok = np.isfinite(M[rows, cols]) & (M[rows, cols] > 0)
    R[rows[num_ok], cols[num_ok]] = 1.0

    # 4) bin along L to a regular grid and average within bins
    L = np.asarray(L_vec, float)
    L_edges = np.arange(l_min, l_max + l_bin_width*0.5, l_bin_width)
    nL = len(L_edges) - 1
    sums   = np.zeros((P, nL), float)
    counts = np.zeros((P, nL), float)

    lbin = np.digitize(L, L_edges) - 1  # 0..nL-1
    for i in range(T):
        j = lbin[i]
        if j < 0 or j >= nL:  # out of range
            continue
        row = R[i, :]
        m = np.isfinite(row)
        if m.any():
            sums[m, j]   += row[m]
            counts[m, j] += 1.0

    Z = sums / np.where(counts > 0, counts, np.nan)  # (P, nL)

    # 5) choose color transform
    if take_log:
        Zc = np.log10(np.clip(Z, EPS_FLOOR, np.inf))
    else:
        Zc = Z

    # 6) pitch edges for pcolormesh
    pitch_edges = pitch_edges_from_centers(pitch)

    return L_edges, pitch_edges, Zc  # ready for pcolormesh (Ny=P, Nx=nL)

# ---------------------------- OMNIweb downloader ---------------------------
def omni_nx1_series(start_dt, end_dt, varcode):
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
                m = re.search(r'(https?://omniweb\.[\w\.]+/staging/[^"\']+\.(?:lst|txt))', r.text, re.I)
                text = requests.get(m.group(1), timeout=60).text if m else r.text

                times, vals = [], []
                for line in text.splitlines():
                    s = line.strip()
                    if not s or s[0] in "#;<":
                        continue
                    parts = s.split()
                    try:
                        if len(parts) >= 3 and parts[0].isdigit() and len(parts[1]) <= 3 and parts[2].isdigit():
                            yyyy, doy, hh = int(parts[0]), int(parts[1]), int(parts[2])
                            dt = datetime(yyyy, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy-1, hours=hh)
                            tail = parts[3:]
                        elif len(parts) >= 4 and all(p.isdigit() for p in parts[:4]):
                            yyyy, mm, dd, hh = map(int, parts[:4])
                            dt = datetime(yyyy, mm, dd, hh, tzinfo=timezone.utc)
                            tail = parts[4:]
                        else:
                            continue
                        if not tail: continue
                        x = tail[-1]
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
    last_err = None
    for var in candidates:
        try:
            s = omni_nx1_series(start_dt, end_dt, var)
            if len(s): return s.sort_index()
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"OMNI: none of {candidates} worked.")

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
    mag_F     = np.concatenate([x["flux_tpe_perkeV"] for x in mag_parts], axis=0)

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
    rept_F     = np.concatenate([x["flux_tpe_perkeV"] for x in rept_parts], axis=0)
    rept_F *= 0.5  # original scaling

    # ---------- series/maps ----------
    mag_log_tl1, mag_how = timeL_at_energy(mag_F, mag_E, mag_A, E_MAG_TL, pitch_deg=PITCH_SELECT_DEG)
    print(f"[MagEIS] Time–L @ {E_MAG_TL} MeV → {mag_how}")

    rept_log_tl2, h2 = timeL_at_energy(rept_F, rept_E, rept_A, E_REPT_TL2, pitch_deg=PITCH_SELECT_DEG)
    rept_log_tl3, h3 = timeL_at_energy(rept_F, rept_E, rept_A, E_REPT_TL3, pitch_deg=PITCH_SELECT_DEG)
    print(f"[REPT]   Time–L @ {E_REPT_TL2} MeV → {h2}")
    print(f"[REPT]   Time–L @ {E_REPT_TL3} MeV → {h3}")

    # ---- Build PAD grids (L×Pitch) with normalization & binning ------------
    L_edges = np.arange(L_MIN, L_MAX + L_BIN_WIDTH*0.5, L_BIN_WIDTH)

    Lm_edges, Pm_edges, Zmag = build_pad_grid(
        mag_F, mag_E, mag_A, mag_L, E_MAG_TL,
        target_deg=90.0, search_halfspan=REF_SEARCH_HALFSPAN,
        max_gap_deg=PITCH_INTERP_MAX_GAP_DEG,
        l_min=L_MIN, l_max=L_MAX, l_bin_width=L_BIN_WIDTH,
        take_log=NORM_PAD_LOG)

    L2_edges, P2_edges, Zr2 = build_pad_grid(
        rept_F, rept_E, rept_A, rept_L, E_REPT_TL2,
        target_deg=90.0, search_halfspan=REF_SEARCH_HALFSPAN,
        max_gap_deg=PITCH_INTERP_MAX_GAP_DEG,
        l_min=L_MIN, l_max=L_MAX, l_bin_width=L_BIN_WIDTH,
        take_log=NORM_PAD_LOG)

    L4_edges, P4_edges, Zr4 = build_pad_grid(
        rept_F, rept_E, rept_A, rept_L, E_REPT_TL3,
        target_deg=90.0, search_halfspan=REF_SEARCH_HALFSPAN,
        max_gap_deg=PITCH_INTERP_MAX_GAP_DEG,
        l_min=L_MIN, l_max=L_MAX, l_bin_width=L_BIN_WIDTH,
        take_log=NORM_PAD_LOG)

    # Time bounds (tz-aware)
    t_mag  = pd.to_datetime(mag_time,  utc=True)
    t_rept = pd.to_datetime(rept_time, utc=True)
    xmin = min(t_mag.min(), t_rept.min())
    xmax = max(t_mag.max(), t_rept.max())

    # ---------- x-axis as "days since start" ----------
    base_day_utc = xmin.floor("D")
    x_mag   = to_day_offset(t_mag,  base_day_utc)
    x_rept  = to_day_offset(t_rept, base_day_utc)
    hours   = hourly_grid_utc(xmin, xmax)
    x_hours = to_day_offset(hours, base_day_utc)

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

    # ---------- OMNI: Kp/Dst ----------
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

    # ------------------------------ PLOT -------------------------------------
    fig, axes = plt.subplots(
        7, 1, figsize=(12, 18), sharex=False,
        gridspec_kw={"height_ratios":[1,1,1,1,1,1,0.9]}
    )
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN,
                        top=TOP_MARGIN, bottom=BOTTOM_MARGIN, hspace=HSPACE)
    fig.suptitle(
        f"{PROBE.upper()} Electron Flux (MagEIS/REPT) — per keV; PAD norm @ 90° (pcolormesh)",
        fontsize=FS_SUP
    )

    extend_kw = 'min' if SHOW_NONPOS_AS_FLOOR else 'neither'
    CBAR_LABEL_FLUX = r"log$_{10}$(cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)"
    if NORM_PAD_LOG:
        vmin_pad, vmax_pad = LOGR_LP_NORM
        CBAR_LABEL_NORM = r"log$_{10}$(Flux / Flux$_{90^\circ}$)"
    else:
        vmin_pad, vmax_pad = LINR_LP_NORM
        CBAR_LABEL_NORM = r"Flux / Flux$_{90^\circ}$"

    # -------- Panels 1–3: Time–L (scatter, log10 color) ---------------------
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
    fig.colorbar(sc1, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw)

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
    fig.colorbar(sc2, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw)

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
    fig.colorbar(sc3, ax=ax, pad=0.01, fraction=0.04, extend=extend_kw)

    fig.text(0.93, 0.70, CBAR_LABEL_FLUX, rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # -------- Panels 4–6: L–Pitch (pcolormesh of normalized PAD) ------------
    # Panel 4: MagEIS 1.079 MeV
    ax = axes[3]
    pm4 = ax.pcolormesh(Lm_edges, Pm_edges, Zmag, shading="auto", cmap="jet",
                        vmin=vmin_pad, vmax=vmax_pad)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlim(L_MIN, L_MAX); ax.set_ylim(0, 180)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_MAG_TL:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    fig.colorbar(pm4, ax=ax, pad=0.01, fraction=0.04)

    # Panel 5: REPT 2.1 MeV
    ax = axes[4]
    pm5 = ax.pcolormesh(L2_edges, P2_edges, Zr2, shading="auto", cmap="jet",
                        vmin=vmin_pad, vmax=vmax_pad)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlim(L_MIN, L_MAX); ax.set_ylim(0, 180)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL2:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    fig.colorbar(pm5, ax=ax, pad=0.01, fraction=0.04)

    # Panel 6: REPT 4.2 MeV
    ax = axes[5]
    pm6 = ax.pcolormesh(L4_edges, P4_edges, Zr4, shading="auto", cmap="jet",
                        vmin=vmin_pad, vmax=vmax_pad)
    ax.set_ylabel("Pitch (deg)"); ax.set_xlim(L_MIN, L_MAX); ax.set_ylim(0, 180)
    ax.text(ENERGY_LABEL_X, 0.5, f"{E_REPT_TL3:g} MeV", transform=ax.transAxes,
            rotation=90, va="center", ha="right", fontsize=FS_LABEL)
    fig.colorbar(pm6, ax=ax, pad=0.01, fraction=0.04)

    fig.text(0.93, 0.37, (r"log$_{10}$(Flux / Flux$_{90^\circ}$)" if NORM_PAD_LOG else r"Flux / Flux$_{90^\circ}$"),
             rotation=90, va="center", ha="center", fontsize=FS_CBAR)

    # -------- Panel 7: Kp & Dst --------------------------------------------
    ax7 = axes[6]
    ax7.grid(True, alpha=0.3, linestyle=":")
    ax7.set_ylabel("Dst (nT)")
    for ax in (axes[0], axes[1], axes[2], ax7):
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.xaxis.set_minor_locator(minor_loc)
        ax.grid(which="minor", alpha=0.15, linestyle=":")
    if kp_dst is not None:
        df = kp_dst.copy().reindex(hours).interpolate(limit_direction="both")
        ax7.plot(x_hours, df["Dst"].values, color="k", lw=1.5, label="Dst")
        ax7r = ax7.twinx()
        ax7r.step(x_hours, df["Kp"].values, where="post", color="tab:orange", lw=1.5, label="Kp")
        ax7r.set_ylabel("Kp", color="tab:orange")
        ax7r.tick_params(axis='y', labelcolor="tab:orange"); ax7r.set_ylim(0, 9)
        lines, labels = [], []
        for axx in (ax7, ax7r):
            h, l = axx.get_legend_handles_labels()
            lines += h; labels += l
        ax7.legend(lines, labels, loc="upper right")
    else:
        ax7.text(0.5,0.5,"No Kp/Dst", transform=ax7.transAxes, ha="center", va="center")
    ax7.set_xlabel("Day (UTC)")

    # Label x for PAD panels
    for ax in (axes[3], axes[4], axes[5]):
        ax.set_xlabel("L")

    fig.savefig(OUT_NAME, dpi=180, bbox_inches="tight", pad_inches=0.02)
    print(f"[SAVED] {OUT_NAME}")
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBSP/REPT: Time × L over a DATE RANGE (3 energies) + Kp/Dst panel, robust tz handling.

v4 changes:
- Force Kp & Dst DataFrames to tz-aware UTC indexes
- Use 'h' minutes/hour codes to avoid FutureWarnings
- Extra debug prints for Kp/Dst (count + timerange)
- Safe text annotation if no Kp/Dst data available
"""

# ========================== USER SETTINGS ==========================
START_DATE = "2015-09-05"              # inclusive, UTC
END_DATE   = "2015-09-15"              # inclusive, UTC

BASE_DIR = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle"
PROBE    = "rbspa"                     # "rbspa" or "rbspb"
RELEASE  = "rel03"
PRODUCT  = "ect-rept-sci-L3"
CHOOSE_LATEST_VERSION = True

FLUX_VAR = None                        # None = auto (prefers FEDU, else FPDU)
PITCH_COMBINE = "median"               # "median" or "mean"
ENERGIES_MEV = [1.8, 3.6, 4.2]

LOG_VMIN, LOG_VMAX = 1.0, 4.0          # log10(flux) range
CMAP = "jet"

SAVE_PLOT = True
SAVE_NAME = f"rept_timeL_{START_DATE}_{END_DATE}_with_kp_dst_v4.png"
# ==================================================================

import os, re
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import cdflib

# ---------------- helpers ----------------
def to_dt_list_any(cdf_epoch_array):
    raw = cdflib.cdfepoch.to_datetime(cdf_epoch_array)
    out = []
    for x in raw:
        ts = pd.Timestamp(x)                      # handles datetime64, datetime, str
        out.append(ts.to_pydatetime().replace(tzinfo=None))  # naive
    return out

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
    if prefer in z: return prefer
    for name in ("FEDU", "FPDU"):
        if name in z: return name
    cands = [v for v in z if any(t in v.lower() for t in ("flux","fedu","fpdu"))]
    cands = [v for v in cands if all(b not in v.lower() for b in ("epoch","energy","alpha","delta","labl"))]
    if not cands: raise RuntimeError("No flux-like variable found. Set FLUX_VAR.")
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
    m = re.search(r"_v(\d+)\.(\d+)\.(\d+)\.cdf$", path)
    return tuple(int(x) for x in m.groups()) if m else (0,0,0)

def find_file_for_date(day: datetime) -> Path | None:
    y = day.strftime("%Y"); ymd = day.strftime("%Y%m%d")
    pattern = f"{PROBE}_{RELEASE}_{PRODUCT}_{ymd}_v*.cdf"
    search_dir = Path(BASE_DIR) / y
    candidates = sorted(glob(str(search_dir / pattern)))
    if not candidates:
        print(f"[MISS] {ymd}: no file in {search_dir}")
        return None
    if CHOOSE_LATEST_VERSION:
        candidates.sort(key=parse_version_from_name)
        chosen = candidates[-1]
    else:
        chosen = candidates[0]
    print(f"[HIT]  {ymd} -> {chosen}")
    return Path(chosen)

def daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ------------- Kp (GFZ first, OMNI fallback) -------------
def fetch_kp_gfz(start_dt, end_dt, status="def"):
    url = "https://kp.gfz.de/app/json/"
    params = {
        "start": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end"  : (end_dt - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "index": "C9",         # Kp*10
        "status": status,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    data = js["data"] if isinstance(js, dict) and "data" in js else js
    times, c9 = [], []
    for item in data:
        if not isinstance(item, dict): continue
        tkey = next((k for k in item if k.lower().startswith("time")), None)
        vkey = next((k for k in item if k.lower() in ("c9","value","kp","ap")), None)
        if tkey and vkey:
            t = pd.Timestamp(item[tkey]).tz_localize("UTC") if pd.Timestamp(item[tkey]).tzinfo is None else pd.Timestamp(item[tkey]).tz_convert("UTC")
            times.append(t)
            c9.append(float(item[vkey]))
    if not times:
        raise RuntimeError("GFZ Kp JSON: no values.")
    df = pd.DataFrame({"Kp": np.asarray(c9)/10.0}, index=pd.DatetimeIndex(times, tz="UTC")).sort_index()
    hourly = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1h", tz="UTC")
    return df.reindex(hourly, method="ffill")

def _fetch_omni_series(start_dt, end_dt, varnum):
    base = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    start_date = start_dt.strftime("%Y%m%d")
    end_date   = (end_dt - timedelta(days=0)).strftime("%Y%m%d")
    params = {
        "activity": "ftp", "res": "hour", "spacecraft": "omni2",
        "start_date": start_date, "end_date": end_date,
        "maxdays": str((end_dt - start_dt).days + 1),
        "vars": varnum, "scale": "Linear", "view": 0, "nsum": 1,
        "paper": 0, "table": 0, "imagex": 640, "imagey": 480,
    }
    r = requests.get(base, params=params, timeout=30); r.raise_for_status()
    m = re.search(r"(https?://omniweb\.gsfc\.nasa\.gov/staging/omni2_[\w\-]+\.lst)", r.text)
    if not m:
        raise RuntimeError("OMNI staging link not found.")
    lst_url = m.group(1)
    r2 = requests.get(lst_url, timeout=30); r2.raise_for_status()
    lines = r2.text.splitlines()
    rows = []
    for line in lines:
        s = line.strip()
        if not s or s[0] in "#;": continue
        parts = s.split()
        try:
            if len(parts) >= 4 and len(parts[1]) <= 3:  # YYYY DOY HH VAL
                yyyy, doy, hh = int(parts[0]), int(parts[1]), int(parts[2])
                val = float(parts[3])
                dt = datetime(yyyy, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy-1, hours=hh)
            elif len(parts) >= 5:                       # YYYY MM DD HH VAL
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
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    vals = np.array([r[1] for r in rows], dtype=float)
    return pd.Series(vals, index=idx).sort_index()

def fetch_kp_omni(start_dt, end_dt):
    s = _fetch_omni_series(start_dt, end_dt, varnum=38)  # Kp*10
    hourly = pd.date_range(start=s.index.min(), end=s.index.max(), freq="1h", tz="UTC")
    return pd.DataFrame({"Kp": (s/10.0).reindex(hourly, method="ffill")})

def fetch_dst_omni(start_dt, end_dt):
    s = _fetch_omni_series(start_dt, end_dt, varnum=40)  # Dst
    hourly = pd.date_range(start=s.index.min(), end=s.index.max(), freq="1h", tz="UTC")
    return pd.DataFrame({"Dst": s.reindex(hourly, method="ffill")})

# ------------- load one REPT CDF -------------
def load_rept(path: Path, prefer_flux=None):
    cdf = cdflib.CDF(str(path))
    flux_name = pick_flux_var(cdf, prefer_flux)
    atts_flux = cdf.varattsget(flux_name)
    fillval = atts_flux.get("FILLVAL", None)
    dep0 = atts_flux.get("DEPEND_0")
    dep1 = atts_flux.get("DEPEND_1")
    dep2 = atts_flux.get("DEPEND_2")

    def cls(name):
        if not name: return "none"
        a = cdf.varattsget(name)
        return classify_axis(name, a)
    cls1, cls2 = cls(dep1), cls(dep2)
    energy_dep = dep1 if cls1 == "energy" else (dep2 if cls2 == "energy" else None)
    pitch_dep  = dep1 if cls1 == "pitch"  else (dep2 if cls2 == "pitch"  else None)
    if energy_dep is None:
        raise RuntimeError(f"{path.name}: cannot identify energy axis.")

    time = to_dt_list_any(cdf.varget(dep0))
    L = (np.squeeze(np.array(cdf.varget("L"), dtype=float))
         if "L" in cdf.cdf_info().zVariables else
         np.squeeze(np.array(cdf.varget("L_star"), dtype=float)) if "L_star" in cdf.cdf_info().zVariables
         else np.full(len(time), np.nan, dtype=float))
    energy_vec = np.squeeze(np.array(cdf.varget(energy_dep), dtype=float))
    units_flux_lbl = istp_units_to_mathtext(atts_flux.get("UNITS",""))

    flux = clean_fill(cdf.varget(flux_name), fillval)

    if flux.ndim == 3:
        time_axis = [i for i, n in enumerate(flux.shape) if n == len(time)]
        time_axis = time_axis[0] if time_axis else 0
        if time_axis != 0: flux = np.moveaxis(flux, time_axis, 0)
        if flux.shape[2] == len(energy_vec):
            pitch_axis, energy_axis = 1, 2
        elif flux.shape[1] == len(energy_vec):
            pitch_axis, energy_axis = 2, 1
        else:
            pitch_axis, energy_axis = 1, 2
        flux_te = np.nanmedian(flux, axis=pitch_axis) if PITCH_COMBINE=="median" else np.nanmean(flux, axis=pitch_axis)
    elif flux.ndim == 2:
        if flux.shape[1] != len(energy_vec): raise RuntimeError(f"{path.name}: 2D flux but NE mismatch.")
        flux_te = flux
    else:
        raise RuntimeError(f"{path.name}: unexpected flux ndim={flux.ndim}")

    return {
        "flux_name": flux_name,
        "units_flux_lbl": units_flux_lbl,
        "time": np.array(time, dtype=object),    # datetimes (naive)
        "L": L,
        "energy_vec": energy_vec,
        "flux_te": np.array(flux_te, dtype=float)
    }

# ---------------------------- MAIN ----------------------------
if __name__ == "__main__":
    # 1) collect REPT files
    start = datetime.fromisoformat(START_DATE)
    end   = datetime.fromisoformat(END_DATE)
    paths = []
    for day in daterange(start, end):
        p = find_file_for_date(day)
        if p is not None and p.exists(): paths.append(p)
    if not paths:
        raise SystemExit("[ABORT] No REPT files in range.")
    print(f"[info] Using {len(paths)} file(s).")

    # 2) load REPT
    parts = []
    for p in paths:
        try: parts.append(load_rept(p, FLUX_VAR))
        except Exception as e: print(f"[SKIP] {p.name}: {e}")
    if not parts:
        raise SystemExit("[ABORT] No usable REPT files.")

    ref_E = parts[0]["energy_vec"]
    idx_list = [int(np.argmin(np.abs(ref_E - E))) for E in ENERGIES_MEV]
    E_used = [float(ref_E[i]) for i in idx_list]
    print(f"[INFO] Energies requested: {ENERGIES_MEV}  -> channels: {E_used}")

    time_all = np.concatenate([p["time"] for p in parts])
    L_all    = np.concatenate([p["L"] for p in parts])

    log_flux_list = []
    for idx in idx_list:
        series = np.concatenate([p["flux_te"][:, idx] for p in parts]).astype(float)
        series[~np.isfinite(series)] = np.nan
        ok = series > 0
        logf = np.full(series.shape, np.nan, dtype=float); logf[ok] = np.log10(series[ok])
        log_flux_list.append(logf)

    flux_name = parts[0]["flux_name"]
    units_flux_lbl = parts[0]["units_flux_lbl"]

    # Build UTC (tz-aware) time axis for plotting & downloads
    x_time = pd.to_datetime(time_all, utc=True)
    start_dt = x_time.min().floor("h").to_pydatetime()
    end_dt   = (x_time.max().ceil("h") + pd.Timedelta(hours=1)).to_pydatetime()

    # 3) Kp (GFZ -> OMNI fallback) and Dst (OMNI)
    kp_df = None
    try:
        kp_df = fetch_kp_gfz(start_dt, end_dt, status="def")
        print(f"[ok] Kp from GFZ: {len(kp_df)} points, {kp_df.index.min()} .. {kp_df.index.max()}")
    except Exception as e:
        print(f"[warn] GFZ Kp failed: {e}")
        try:
            kp_df = fetch_kp_omni(start_dt, end_dt)
            print(f"[ok] Kp from OMNI: {len(kp_df)} points, {kp_df.index.min()} .. {kp_df.index.max()}")
        except Exception as e2:
            print(f"[warn] OMNI Kp failed: {e2}")
            kp_df = None

    dst_df = None
    try:
        dst_df = fetch_dst_omni(start_dt, end_dt)
        print(f"[ok] Dst from OMNI: {len(dst_df)} points, {dst_df.index.min()} .. {dst_df.index.max()}")
    except Exception as e:
        print(f"[warn] Dst download failed: {e}")
        dst_df = None

    # 4) FIGURE
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1,1,1,0.6]})
    fig.suptitle(f"{PROBE.upper()}  Time–L flux (pitch-{PITCH_COMBINE}) + Kp/Dst\n"
                 f"{START_DATE} to {END_DATE} (UTC)")

    # Flux panels
    for ax, logf, E in zip(axes[:3], log_flux_list, E_used):
        ok = np.isfinite(logf)
        sc = ax.scatter(x_time[ok], L_all[ok], c=logf[ok], s=4, cmap=CMAP,
                        vmin=LOG_VMIN, vmax=LOG_VMAX)
        ax.set_ylabel("L-shell")
        ax.set_title(f"{flux_name}: L vs Time @ ~{E:.2f} MeV")

    # slim colorbar
    fig.subplots_adjust(right=0.86, hspace=0.32)
    cbar_ax = fig.add_axes([0.88, 0.17, 0.02, 0.63])
    fig.colorbar(sc, cax=cbar_ax, label="log$_{10}$ " + (units_flux_lbl or "Flux"))

    # Kp/Dst panel
    ax4 = axes[3]
    ax4.set_title("Dst (left) and Kp (right)")
    ax4.grid(True, alpha=0.3, linestyle=":")
    ax4.set_ylabel("Dst (nT)")

    hours = pd.date_range(start=x_time.min().floor("h"), end=x_time.max().ceil("h"),
                          freq="1h", tz="UTC")

    plotted_any = False
    if dst_df is not None and not dst_df.empty:
        # ensure tz-aware UTC
        dst_df.index = pd.to_datetime(dst_df.index, utc=True)
        dst_plot = dst_df.reindex(hours).interpolate(limit_direction="both")
        if dst_plot["Dst"].notna().any():
            ax4.plot(dst_plot.index, dst_plot["Dst"], lw=1.5, color="k", label="Dst")
            plotted_any = True

    if kp_df is not None and not kp_df.empty:
        kp_df.index = pd.to_datetime(kp_df.index, utc=True)
        kp_plot = kp_df.reindex(hours).interpolate(limit_direction="both")
        if kp_plot["Kp"].notna().any():
            ax4r = ax4.twinx()
            ax4r.step(kp_plot.index, kp_plot["Kp"], where="post",
                      lw=1.5, color="tab:orange", label="Kp")
            ax4r.set_ylabel("Kp", color="tab:orange")
            ax4r.tick_params(axis='y', labelcolor="tab:orange")
            ax4r.set_ylim(0, 9)
            # joint legend
            lines, labels = [], []
            for axx in (ax4, ax4r):
                h, l = axx.get_legend_handles_labels()
                lines += h; labels += l
            ax4.legend(lines, labels, loc="upper right")
            plotted_any = True

    if not plotted_any:
        ax4.text(0.5, 0.5, "No Kp/Dst data available\n(check internet or service availability)",
                 transform=ax4.transAxes, ha="center", va="center", fontsize=11)

    axes[-1].set_xlabel("Time (UTC)")
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H", tz=timezone.utc))

    if SAVE_PLOT:
        fig.savefig(SAVE_NAME, dpi=180)
        print(f"[SAVED] {SAVE_NAME}")

    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RBSP/REPT time–L plots at pitch≈90° over a date range + Kp/Dst
Now with a unit switch:
    ENERGY_UNIT = "MeV" or "keV"
    ENERGIES is interpreted in that unit; flux is per that unit.
"""

# ============================== USER SETTINGS ==============================
START_DATE = "2017-05-27"          # inclusive, UTC
END_DATE   = "2017-05-28"          # inclusive, UTC

BASE_DIR = "/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle"
PROBE    = "rbspa"                 # "rbspa" or "rbspb"
RELEASE  = "rel03"
PRODUCT  = "ect-rept-sci-L3"
CHOOSE_LATEST_VERSION = True       # pick newest v*.cdf if multiple for a day

FLUX_VAR = None                    # None=auto (prefers "FEDU", else "FPDU")

PITCH_SELECT_DEG = 90.0            # choose the pitch bin nearest to this angle

# ---- NEW: choose energy unit and list energies in that unit ----
# ENERGY_UNIT = "MeV"                # "MeV" or "keV" (case-insensitive)
# ENERGIES    = [1.8, 3.4, 4.2]      # interpreted in ENERGY_UNIT

ENERGY_UNIT = "keV"
ENERGIES    = [1800, 3400, 4200]   # keV

# Exact channel only? (False = do log-linear interpolation if needed)
EXACT_ONLY  = False

# Plot style (values are ranges for log10 of the chosen per-unit flux)
LOG_VMIN, LOG_VMAX = 1.0, 4.0
CMAP = "jet"

SAVE_PLOT = True
SAVE_NAME = f"rept_timeL_pitch90_{ENERGY_UNIT}_2.png"
# ==========================================================================

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

# ------------------------------ utilities ---------------------------------
def to_dt_list_any(cdf_epoch_array):
    raw = cdflib.cdfepoch.to_datetime(cdf_epoch_array)
    out = []
    for x in raw:
        ts = pd.Timestamp(x)  # handles datetime, datetime64, str
        out.append(ts.to_pydatetime().replace(tzinfo=None))
    return out

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

def find_file_for_date(day: datetime) -> Path | None:
    y = day.strftime("%Y"); ymd = day.strftime("%Y%m%d")
    pattern = f"{PROBE}_{RELEASE}_{PRODUCT}_{ymd}_v*.cdf"
    folder = Path(BASE_DIR) / y
    matches = sorted(glob(str(folder / pattern)))
    if not matches:
        print(f"[MISS] {ymd}: no file under {folder}")
        return None
    if CHOOSE_LATEST_VERSION:
        matches.sort(key=parse_version_from_name)
        chosen = matches[-1]
    else:
        chosen = matches[0]
    print(f"[HIT]  {ymd} → {chosen}")
    return Path(chosen)

def daterange(start: datetime, end: datetime):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ------------------------------ OMNI downloads ----------------------------
def _fetch_omni_series(start_dt, end_dt, varnum):
    base = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
    params = {
        "activity": "ftp", "res": "hour", "spacecraft": "omni2",
        "start_date": start_dt.strftime("%Y%m%d"),
        "end_date":   (end_dt - timedelta(days=0)).strftime("%Y%m%d"),
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

def fetch_kp_omni(start_dt, end_dt):
    s = _fetch_omni_series(start_dt, end_dt, 38)  # Kp*10
    hourly = pd.date_range(start=s.index.min(), end=s.index.max(), freq="1h", tz="UTC")
    return pd.DataFrame({"Kp": (s/10.0).reindex(hourly, method="ffill")})

def fetch_dst_omni(start_dt, end_dt):
    s = _fetch_omni_series(start_dt, end_dt, 40)  # Dst
    hourly = pd.date_range(start=s.index.min(), end=s.index.max(), freq="1h", tz="UTC")
    return pd.DataFrame({"Dst": s.reindex(hourly, method="ffill")})

# ------------------------------- REPT loader ------------------------------
def load_rept_day(path: Path, prefer_flux=None, pitch_target_deg=90.0, output_unit="MeV"):
    """
    Load one daily CDF. Returns dict with:
      time, L (cleaned), energy_vec_MeV, flux_te_per_unit (time x energy)
    output_unit: "MeV" (no conversion) or "keV" (divide by 1000)
    """
    cdf = cdflib.CDF(str(path))

    # flux var
    flux_name = pick_flux_var(cdf, prefer_flux)
    atts_flux = cdf.varattsget(flux_name)
    fv_flux   = atts_flux.get("FILLVAL", None)
    dep0 = atts_flux.get("DEPEND_0")
    dep1 = atts_flux.get("DEPEND_1")
    dep2 = atts_flux.get("DEPEND_2")

    # classify axes
    cls1 = classify_axis(dep1, cdf)
    cls2 = classify_axis(dep2, cdf)
    energy_dep = dep1 if cls1 == "energy" else (dep2 if cls2 == "energy" else None)
    pitch_dep  = dep1 if cls1 == "pitch"  else (dep2 if cls2 == "pitch"  else None)
    if energy_dep is None:
        raise RuntimeError(f"{path.name}: cannot identify energy axis.")

    # time
    time = to_dt_list_any(cdf.varget(dep0))

    # L (cleaned)
    if "L" in cdf.cdf_info().zVariables:
        L = clean_fill(cdf.varget("L"), cdf.varattsget("L").get("FILLVAL", None))
    elif "L_star" in cdf.cdf_info().zVariables:
        L = clean_fill(cdf.varget("L_star"), cdf.varattsget("L_star").get("FILLVAL", None))
    else:
        L = np.full(len(time), np.nan, dtype=float)
    with np.errstate(invalid="ignore"):
        L[(L < 0.5) | (L > 10.0)] = np.nan
    L = np.squeeze(L).astype(float)

    # energy vector (MeV in file)
    energy_vec_MeV = np.squeeze(np.array(cdf.varget(energy_dep), dtype=float))

    # pitch vector (deg)
    if pitch_dep is None:
        raise RuntimeError(f"{path.name}: pitch axis missing.")
    try:
        pitch_vec = np.squeeze(np.array(cdf.varget(pitch_dep), dtype=float))
    except Exception:
        raise RuntimeError(f"{path.name}: pitch vector not readable.")

    # flux array
    flux = clean_fill(cdf.varget(flux_name), fv_flux)

    # reorder to (time, pitch, energy) and select pitch≈target
    if flux.ndim == 3:
        time_axis = [i for i, n in enumerate(flux.shape) if n == len(time)]
        time_axis = time_axis[0] if time_axis else 0
        if time_axis != 0:
            flux = np.moveaxis(flux, time_axis, 0)

        if flux.shape[2] == len(energy_vec_MeV):
            pitch_axis, energy_axis = 1, 2
        elif flux.shape[1] == len(energy_vec_MeV):
            pitch_axis, energy_axis = 2, 1
        else:
            pitch_axis, energy_axis = 1, 2

        pv = np.array(pitch_vec, dtype=float).ravel()
        dist = np.minimum(np.abs(pv - pitch_target_deg),
                          np.abs((pv % 360.0) - pitch_target_deg))
        pidx = int(np.nanargmin(dist))
        flux_te = np.take(flux, indices=pidx, axis=pitch_axis)  # (time, energy)
    elif flux.ndim == 2:
        if flux.shape[1] != len(energy_vec_MeV):
            raise RuntimeError(f"{path.name}: 2D flux but NE mismatch.")
        flux_te = flux
    else:
        raise RuntimeError(f"{path.name}: unexpected flux ndim={flux.ndim}")

    # Convert per MeV → per chosen unit
    unit = output_unit.lower()
    if unit == "kev":
        flux_te_per_unit = np.array(flux_te, dtype=float) / 1000.0
    elif unit == "mev":
        flux_te_per_unit = np.array(flux_te, dtype=float)
    else:
        raise ValueError("output_unit must be 'MeV' or 'keV'")

    return dict(
        time=np.array(time, dtype=object),
        L=L,
        energy_vec_MeV=energy_vec_MeV,
        flux_te_per_unit=flux_te_per_unit,
        flux_name=flux_name,
    )

# -------------------------- energy selection ------------------------------
def concat_flux(parts):
    return np.concatenate([p["flux_te_per_unit"] for p in parts], axis=0)

def series_at_specific_energies(flux_te_per_unit, energy_vec_MeV, energies_req_MeV,
                                tol=1e-6, exact_only=False):
    """
    For each requested E (MeV), return a time series at *that energy*:
      - exact channel center (within tol): take that column
      - else: log-linear interpolation across energy (unless exact_only=True)
    Returns: list of 1D arrays, and meta [(Ereq, 'exact' or 'interp e0–e1 <unit>'), ...]
    """
    E = np.asarray(energy_vec_MeV, dtype=float)
    series_list, meta = [], []

    for Ereq in energies_req_MeV:
        exact_idx = np.where(np.isclose(E, Ereq, rtol=0.0, atol=tol))[0]
        if exact_idx.size > 0:
            col = flux_te_per_unit[:, exact_idx[0]].astype(float)
            series_list.append(col)
            meta.append((Ereq, "exact"))
            continue

        if exact_only:
            raise ValueError(f"Requested energy {Ereq} MeV is not an exact channel center.")

        if not (E.min() <= Ereq <= E.max()):
            raise ValueError(f"Requested energy {Ereq} MeV out of range {E.min()}–{E.max()} MeV")

        hi = int(np.searchsorted(E, Ereq, side="right"))
        lo = hi - 1
        e0, e1 = E[lo], E[hi]
        w = (Ereq - e0) / (e1 - e0)

        f0 = flux_te_per_unit[:, lo].astype(float)
        f1 = flux_te_per_unit[:, hi].astype(float)

        out = np.full_like(f0, np.nan, dtype=float)
        m = (f0 > 0) & (f1 > 0)
        out[m] = np.exp(np.log(f0[m]) * (1.0 - w) + np.log(f1[m]) * w)
        left_only  = (f0 > 0) & ~np.isfinite(f1)
        right_only = (f1 > 0) & ~np.isfinite(f0)
        out[left_only]  = f0[left_only]
        out[right_only] = f1[right_only]

        series_list.append(out)
        meta.append((Ereq, f"interp {e0:g}–{e1:g} MeV"))

    return series_list, meta

# ----------------------------------- MAIN ----------------------------------
if __name__ == "__main__":
    # Normalize unit flag
    ENERGY_UNIT = ENERGY_UNIT.strip()
    unit_lower = ENERGY_UNIT.lower()
    if unit_lower not in ("mev", "kev"):
        raise SystemExit("ENERGY_UNIT must be 'MeV' or 'keV'.")

    # Energies requested → MeV (internal)
    if unit_lower == "keV":
        energies_req_MeV = [float(e)/1000.0 for e in ENERGIES]
    else:
        energies_req_MeV = [float(e) for e in ENERGIES]

    # Collect files
    start = datetime.fromisoformat(START_DATE)
    end   = datetime.fromisoformat(END_DATE)
    paths = []
    for d in daterange(start, end):
        p = find_file_for_date(d)
        if p is not None and p.exists():
            paths.append(p)
    if not paths:
        raise SystemExit("[ABORT] No REPT files found in the requested range.")
    print(f"[info] Using {len(paths)} file(s).")

    # Load days (pitch≈90°, per chosen unit)
    parts = []
    for p in paths:
        try:
            parts.append(load_rept_day(
                p, FLUX_VAR, pitch_target_deg=PITCH_SELECT_DEG, output_unit=ENERGY_UNIT
            ))
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
    if not parts:
        raise SystemExit("[ABORT] No usable REPT files.")

    # Concatenate time/L/flux
    time_all = np.concatenate([p["time"] for p in parts])
    L_all    = np.concatenate([p["L"]   for p in parts])
    flux_all = concat_flux(parts)
    energy_vec_MeV = parts[0]["energy_vec_MeV"]
    flux_name      = parts[0]["flux_name"]

    # Build series at *specific* energies (exact or interpolated)
    flux_series_list, energy_meta = series_at_specific_energies(
        flux_all, energy_vec_MeV, energies_req_MeV, tol=1e-6, exact_only=EXACT_ONLY
    )

    # Convert to log10 (mask non-positive)
    log_flux_list = []
    for ser in flux_series_list:
        s = ser.astype(float)
        s[~np.isfinite(s)] = np.nan
        ok = s > 0
        logf = np.full(s.shape, np.nan, dtype=float)
        logf[ok] = np.log10(s[ok])
        log_flux_list.append(logf)

    # tz-aware times for plotting & indices
    x_time = pd.to_datetime(time_all, utc=True)

    # ---- Kp/Dst from OMNI (optional) ----
    start_dt = x_time.min().floor("h").to_pydatetime()
    end_dt   = (x_time.max().ceil("h") + pd.Timedelta(hours=1)).to_pydatetime()

    kp_df = None
    dst_df = None
    try:
        kp_df = fetch_kp_omni(start_dt, end_dt)
        print(f"[ok] Kp downloaded: {len(kp_df)} points")
    except Exception as e:
        print(f"[warn] Kp download failed: {e}")
    try:
        dst_df = fetch_dst_omni(start_dt, end_dt)
        print(f"[ok] Dst downloaded: {len(dst_df)} points")
    except Exception as e:
        print(f"[warn] Dst download failed: {e}")

    # ------------------------------ PLOT --------------------------------
    unit_label = "keV" if unit_lower == "keV" else "MeV"
    energy_label_values = (
        [int(round(e*1000)) for e in energies_req_MeV] if unit_lower=="keV"
        else [e for e in energies_req_MeV]
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1,1,1,0.6]})
    fig.suptitle(f"{PROBE.upper()}  Time–L flux (pitch≈{PITCH_SELECT_DEG:.0f}°) per {unit_label} + Kp/Dst\n"
                 f"{START_DATE} to {END_DATE} (UTC)")

    # Panels 1–3: time–L scatter colored by log10 flux per chosen unit
    for ax, logf, (Ereq_MeV, how), E_disp in zip(
        axes[:3], log_flux_list, energy_meta, energy_label_values
    ):
        ok = np.isfinite(logf) & np.isfinite(L_all)
        sc = ax.scatter(x_time[ok], L_all[ok], c=logf[ok], s=4,
                        cmap=CMAP, vmin=LOG_VMIN, vmax=LOG_VMAX)
        if np.any(ok):
            lo, hi = np.nanpercentile(L_all[ok], [1, 99])
            lo = max(0.5, lo); hi = min(10.0, hi)
            if hi - lo < 0.5: lo, hi = 1.0, 7.0
            ax.set_ylim(lo, hi)
        else:
            ax.set_ylim(1.0, 7.0)
        ax.set_ylabel("L-shell")
        ax.set_title(f"L vs Time @ {E_disp:g} {unit_label}  ({how})")

    # Slim colorbar on the right
    fig.subplots_adjust(right=0.86, hspace=0.32)
    cax = fig.add_axes([0.88, 0.17, 0.02, 0.63])
    fig.colorbar(sc, cax=cax,
                 label=rf"log$_{{10}}$ (cm$^{{-2}}$ s$^{{-1}}$ sr$^{{-1}}$ {unit_label}$^{{-1}}$)")

    # Panel 4: Dst (left) and Kp (right)
    ax4 = axes[3]
    ax4.set_title("Dst (left) and Kp (right)")
    ax4.grid(True, alpha=0.3, linestyle=":")
    ax4.set_ylabel("Dst (nT)")

    hours = pd.date_range(start=x_time.min().floor("h"),
                          end=x_time.max().ceil("h"),
                          freq="1h", tz="UTC")

    plotted_any = False
    if dst_df is not None and not dst_df.empty:
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
            lines, labels = [], []
            for axx in (ax4, ax4r):
                h, l = axx.get_legend_handles_labels()
                lines += h; labels += l
            ax4.legend(lines, labels, loc="upper right")
            plotted_any = True

    if not plotted_any:
        ax4.text(0.5, 0.5, "No Kp/Dst data", transform=ax4.transAxes,
                 ha="center", va="center", fontsize=11)

    axes[-1].set_xlabel("Time (UTC)")
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H", tz=timezone.utc))

    if SAVE_PLOT:
        fig.savefig(SAVE_NAME, dpi=180)
        print(f"[SAVED] {SAVE_NAME}")

    plt.show(block=True)

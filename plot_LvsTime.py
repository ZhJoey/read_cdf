import numpy as np
import matplotlib.pyplot as plt
import cdflib

# Load CDF
cdf_file = cdflib.CDF("/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle/2017/rbspa_rel03_ect-rept-sci-L3_20170528_v5.3.0.cdf")
flux = cdf_file.varget("FEDU")  # shape: time, pitch, energy
energy = cdf_file.varget("FEDU_Energy")
L_shell = cdf_file.varget("L")
time = cdflib.cdfepoch.to_datetime(cdf_file.varget("Epoch"))

# Pick a single energy channel (example: closest to 1.8 MeV)
energy_idx = np.argmin(np.abs(energy - 1.8))
flux_energy = flux[:, :, energy_idx]

# Average over pitch angles
flux_avg_pitch = np.nanmedian(flux_energy, axis=1)

# Mask invalid values
fillval = cdf_file.varattsget("FEDU")["FILLVAL"]
flux_avg_pitch = np.where(flux_avg_pitch == fillval, np.nan, flux_avg_pitch)

# Plot
plt.figure(figsize=(10,5))
sc = plt.scatter(time, L_shell, c=np.log10(flux_avg_pitch),
                 cmap="jet", s=2, vmin=0, vmax=2) # adjust vmin/vmax to your range
cbar = plt.colorbar(sc, label="log10 flux [#/cm²/s/sr/keV]")
plt.ylabel("L-shell")
plt.xlabel("Time (UTC)")
plt.title("Flux vs L-shell (pitch-averaged, ~1.8 MeV)")
plt.savefig("LvsTime1")

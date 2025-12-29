import cdflib

# Load the file
cdf_file = cdflib.CDF("/export/sec15-data/data/rbm-data/RBSP/OriginalData/rbspa/rept/level3/pitchangle/2017/rbspa_rel03_ect-rept-sci-L3_20170528_v5.3.0.cdf")

# See what variables exist
print(cdf_file.cdf_info())

# Get variable names
variables = cdf_file.cdf_info().zVariables
print("Variables:", variables)

# Read a specific variable (e.g., time)
time = cdf_file.varget("Epoch")

# Convert CDF Epoch to datetime
from cdflib.epochs import CDFepoch

flux = cdf_file.varget("FEDU")  # electron flux
PitchAngle = cdf_file.varget("FEDU_Alpha")
energy = cdf_file.varget("FEDU_Energy")
time_cdf = cdf_file.varget("Epoch")
time_dt = CDFepoch.to_datetime(time_cdf)

print("Flux shape:", flux.shape)
print("PitchAngle bins:", PitchAngle)
print("Energy bins:", energy)
print("First 5 timestamps:", time_dt[:5])

# src/parameters.py

# ---------------------------------------------------------
# QCCD Operation Durations (Unit: microseconds)
# Based on Source 1, Table 1 
# ---------------------------------------------------------

# Gate Operations (LIZ)
DURATION_MS = 40.0      # Two-qubit MS gate
DURATION_R = 5.0        # Single-qubit Rotation (Rx, Ry, Rz)
DURATION_MEASURE = 1000.0
DURATION_RESET = 50.0

# Movement Operations (Movement)
DURATION_SHUTTLE = 5.0  # 1 hop movement (approx)
DURATION_SPLIT = 100.0
DURATION_MERGE = 100.0
DURATION_SWAP = 20.0    # Costly operation in 1D
"""
Generate Baseline 2 (QCCD-Aware / Human Scheduled) Circuit
人間がスケジューリングした「衝突のない4ステップ実行」をStim形式で生成する。
"""

import os

def generate_baseline_aware():
    # Helper to generate CNOT(c, t) native gates for QCCD DSL
    def cnot_dsl(c, t):
        s = f"# CNOT({c}, {t}) - Step Group\n"
        s += f"R {c} 1.5708 1.5708\n"
        s += f"MS {c} {t} 0.25\n"
        s += f"R {c} 1.5708 3.1416\n"
        s += f"R {t} 1.5708 3.1416\n"
        s += f"R {c} 1.5708 4.7124\n"
        return s

    # --------------------------------------------------------
    # Human-Optimized Schedule (4 Steps)
    # 各ステップ内でデータ量子ビットの衝突がないようにペアリング
    # --------------------------------------------------------
    
    # Step 1: 
    # X-Ancillas: 9->1, 10->2, 11->5, 12->7
    # Z-Ancillas: 13->0, 14->4, 15->3, 16->8
    step1_pairs = [
        (9, 1), (10, 2), (11, 5), (12, 7), # X stabilizers
        (13, 0), (14, 4), (15, 3), (16, 8) # Z stabilizers (Control->Target for CNOT logic)
    ]
    # Note: Z stabilizer measurement usually involves CNOT(Data, Ancilla).
    # However, in standard QCCD/Maslov decomposition, the direction of MS gate 
    # and basis rotation determines logic.
    # For Stim compatibility in this project, we assume the input format 
    # uses a consistent "CNOT Control Target" notation where:
    #   X-Stab: Ancilla is Control, Data is Target (Ancilla in + state)
    #   Z-Stab: Data is Control, Ancilla is Target (Ancilla in 0 state)
    # 
    # *CRITICAL CHECK*: 
    # Your system seems to define CNOT(c,t) macro universally.
    # For Z-stabilizers (measuring Z parity), we need Data->Ancilla CNOTs.
    # So for Z-ancillas (13-16), the pairs should be (Data, Ancilla).
    
    step1_ops = [
        (9, 1), (10, 2), (11, 5), (12, 7),  # X: Ancilla -> Data
        (0, 13), (4, 14), (3, 15), (8, 16)  # Z: Data -> Ancilla
    ]

    # Step 2:
    # X: 9->0, 11->4, 12->6
    # Z: 14->1, 15->7, 16->5
    step2_ops = [
        (9, 0), (11, 4), (12, 6),
        (1, 14), (7, 15), (5, 16)
    ]

    # Step 3:
    # X: 9->4, 11->8
    # Z: 13->3, 14->2, 15->6
    step3_ops = [
        (9, 4), (11, 8),
        (3, 13), (2, 14), (6, 15)
    ]

    # Step 4:
    # X: 9->3, 11->7
    # Z: 14->5, 15->4
    step4_ops = [
        (9, 3), (11, 7),
        (5, 14), (4, 15)
    ]

    # --------------------------------------------------------
    # Build Circuit String
    # --------------------------------------------------------
    
    circuit_body = ""
    
    # 1. Initialization (Hadamard on X ancillas: 9, 10, 11, 12)
    #    Simulated by Ry(pi/2) for X-basis prep if needed, or just R gates.
    #    Your system uses explicit R gates for H.
    circuit_body += "# --- Initialization (H on X ancillas) ---\n"
    for anc in [9, 10, 11, 12]:
        # H = R(pi, pi/2)? Or R(pi/2, 0) + ... 
        # Using the macro from your files: R q 1.5708 0.0; R q 1.5708 1.5708
        circuit_body += f"R {anc} 3.1416 0.0\n" 
        circuit_body += f"R {anc} 1.5708 1.5708\n"

    # 2. CNOT Cycles
    circuit_body += "\n# --- Step 1: Parallel Interaction Group A ---\n"
    for c, t in step1_ops:
        circuit_body += cnot_dsl(c, t)

    circuit_body += "\n# --- Step 2: Parallel Interaction Group B ---\n"
    for c, t in step2_ops:
        circuit_body += cnot_dsl(c, t)

    circuit_body += "\n# --- Step 3: Parallel Interaction Group C ---\n"
    for c, t in step3_ops:
        circuit_body += cnot_dsl(c, t)

    circuit_body += "\n# --- Step 4: Parallel Interaction Group D ---\n"
    for c, t in step4_ops:
        circuit_body += cnot_dsl(c, t)

    # 3. Measurement (H on X ancillas, then Measure All)
    circuit_body += "\n# --- Measurement Prep (H on X ancillas) ---\n"
    for anc in [9, 10, 11, 12]:
        circuit_body += f"R {anc} 3.1416 0.0\n"
        circuit_body += f"R {anc} 1.5708 1.5708\n"
    
    circuit_body += "\n# --- Readout ---\n"
    for anc in range(9, 17):
        circuit_body += f"MEASURE {anc}\n"

    # --------------------------------------------------------
    # Output to File
    # --------------------------------------------------------
    file_content = f"""
# ============================================================
# Surface-17 Syndrome Extraction - Baseline 2 (Human Scheduled)
# Standard 4-Step Parallel Schedule
# ============================================================

QUBIT_COUNT = 17
DATA_QUBITS = list(range(9))
ANCILLA_QUBITS = list(range(9, 17))
DISTANCE = 3

CIRCUIT = \"\"\"
{circuit_body}
\"\"\"
"""
    
    os.makedirs("circuits", exist_ok=True)
    with open("circuits/baseline_aware_d3.py", "w") as f:
        f.write(file_content.strip())

    print("Successfully generated: circuits/baseline_aware_d3.py")

if __name__ == "__main__":
    generate_baseline_aware()
"""
Distance-5 Rotated Surface Code 回路生成器

Data qubits: 0-24 (5x5 grid)
   0  1  2  3  4
   5  6  7  8  9
  10 11 12 13 14
  15 16 17 18 19
  20 21 22 23 24

Stabilizers: 24 (12 X-type + 12 Z-type)

Hook error avoiding CNOT order:
  X stabilizers (Z-pattern): NW -> NE -> SW -> SE
  Z stabilizers (reverse-N): NW -> SW -> NE -> SE

使い方:
  python d5_generator.py           # reuse1とreuse2を生成
"""

D = 5
N_DATA = D * D  # 25


def qubit_id(r, c):
    return r * D + c


# ==========================================
# Stabilizer Definitions
# ==========================================

def get_stabilizers():
    """d=5 rotated surface code の全スタビライザを返す"""
    stabilizers = []

    # Bulk plaquettes (4x4 grid)
    for pr in range(D - 1):
        for pc in range(D - 1):
            nw = qubit_id(pr, pc)
            ne = qubit_id(pr, pc + 1)
            sw = qubit_id(pr + 1, pc)
            se = qubit_id(pr + 1, pc + 1)

            if (pr + pc) % 2 == 0:
                # X-type (Z-pattern): NW -> NE -> SW -> SE
                stabilizers.append(("X", [nw, ne, sw, se], (pr, pc)))
            else:
                # Z-type (reverse-N): NW -> SW -> NE -> SE
                stabilizers.append(("Z", [nw, sw, ne, se], (pr, pc)))

    # Boundary stabilizers
    # Left boundary (Z-type, weight 2)
    for pr in range(D - 1):
        if (pr + (-1)) % 2 == 1:  # Z-type at odd parity
            n = qubit_id(pr, 0)
            s = qubit_id(pr + 1, 0)
            stabilizers.append(("Z", [n, s], ("left", pr)))

    # Right boundary (Z-type, weight 2)
    for pr in range(D - 1):
        if (pr + (D - 1)) % 2 == 1:  # Z-type at odd parity
            n = qubit_id(pr, D - 1)
            s = qubit_id(pr + 1, D - 1)
            stabilizers.append(("Z", [n, s], ("right", pr)))

    # Top boundary (X-type, weight 2)
    for pc in range(D - 1):
        if ((-1) + pc) % 2 == 0:  # X-type at even parity
            w = qubit_id(0, pc)
            e = qubit_id(0, pc + 1)
            stabilizers.append(("X", [w, e], ("top", pc)))

    # Bottom boundary (X-type, weight 2)
    for pc in range(D - 1):
        if ((D - 1) + pc) % 2 == 0:  # X-type at even parity
            w = qubit_id(D - 1, pc)
            e = qubit_id(D - 1, pc + 1)
            stabilizers.append(("X", [w, e], ("bottom", pc)))

    return stabilizers


def get_snake_order(stabilizers):
    """蛇行順でスタビライザを並べ替える

    Bulk plaquettes を蛇行順に並べ、boundary stabilizers を適切な位置に挿入
    """
    bulk = [s for s in stabilizers if isinstance(s[2], tuple) and isinstance(s[2][0], int)]
    left_boundary = [s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "left"]
    right_boundary = [s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "right"]
    top_boundary = [s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "top"]
    bottom_boundary = [s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "bottom"]

    # Bulk を行ごとにグループ化
    bulk_by_row = {}
    for s in bulk:
        pr, pc = s[2]
        bulk_by_row.setdefault(pr, []).append((pc, s))

    # 蛇行順
    ordered = []
    for pr in range(D - 1):
        row = sorted(bulk_by_row.get(pr, []), key=lambda x: x[0])
        if pr % 2 == 1:
            row = list(reversed(row))

        # 行の先頭に左boundary / 末尾に右boundary を挿入
        if pr % 2 == 0:
            # 左→右: 左boundary → bulk → 右boundary
            for lb in left_boundary:
                if lb[2][1] == pr:
                    ordered.append(lb)
            for _, s in row:
                ordered.append(s)
            for rb in right_boundary:
                if rb[2][1] == pr:
                    ordered.append(rb)
        else:
            # 右→左: 右boundary → bulk(reverse) → 左boundary
            for rb in right_boundary:
                if rb[2][1] == pr:
                    ordered.append(rb)
            for _, s in row:
                ordered.append(s)
            for lb in left_boundary:
                if lb[2][1] == pr:
                    ordered.append(lb)

        # 行の終わりに top/bottom boundary を挿入
        if pr == 0:
            # Top boundary: 行の最後の列付近に挿入
            for tb in sorted(top_boundary, key=lambda s: s[2][1]):
                ordered.append(tb)
        if pr == D - 2:
            # Bottom boundary: 最終行の後に
            for bb in sorted(bottom_boundary, key=lambda s: s[2][1]):
                ordered.append(bb)

    return ordered


# ==========================================
# Circuit Generation
# ==========================================

def z_cnot_block(data, ancilla):
    """Z-type CNOT decomposition"""
    return [
        f"R {data} 1.5708 1.5708",
        f"MS {data} {ancilla} 0.25",
        f"R {data} 1.5708 3.1416",
        f"R {ancilla} 1.5708 3.1416",
        f"R {data} 1.5708 4.7124",
    ]


def x_cnot_block(ancilla, data):
    """X-type CNOT decomposition"""
    return [
        f"R {ancilla} 1.5708 1.5708",
        f"MS {ancilla} {data} 0.25",
        f"R {ancilla} 1.5708 3.1416",
        f"R {data} 1.5708 3.1416",
        f"R {ancilla} 1.5708 4.7124",
    ]


def x_init(ancilla):
    """X-type stabilizer initialization (Hadamard-like)"""
    return [
        f"R {ancilla} 3.1416 0.0",
        f"R {ancilla} 1.5708 1.5708",
    ]


def x_finalize(ancilla):
    """X-type stabilizer finalization"""
    return [
        f"R {ancilla} 3.1416 0.0",
        f"R {ancilla} 1.5708 1.5708",
    ]


def generate_stabilizer_block(stab_type, data_qubits, ancilla):
    """1スタビライザの命令列を生成"""
    lines = []
    lines.append(f"RESET {ancilla}")

    if stab_type == "Z":
        for dq in data_qubits:
            lines.extend(z_cnot_block(dq, ancilla))
    else:  # X
        lines.extend(x_init(ancilla))
        for dq in data_qubits:
            lines.extend(x_cnot_block(ancilla, dq))
        lines.extend(x_finalize(ancilla))

    lines.append(f"MEASURE {ancilla}")
    return lines


def generate_circuit(n_ancillas=1):
    """完全な回路文字列を生成

    Args:
        n_ancillas: 使用するアンシラ数 (1 or 2)
    """
    stabilizers = get_stabilizers()
    ordered = get_snake_order(stabilizers)

    n_data = N_DATA
    ancilla_ids = list(range(n_data, n_data + n_ancillas))

    lines = []
    lines.append("# ============================================================")
    lines.append(f"# Distance-5 Rotated Surface Code (reuse{n_ancillas})")
    lines.append("# ============================================================")
    lines.append("")

    # Qubit declarations
    lines.append("# Qubit Declarations")
    for i in range(n_data):
        r, c = divmod(i, D)
        lines.append(f"QUBIT {i} AT ({r}, {c})")
    for i, aid in enumerate(ancilla_ids):
        lines.append(f"QUBIT {aid} AT (10, {i})")
    lines.append("")

    # Instructions
    lines.append("# Instructions")
    stab_info = []
    for idx, (stype, dqubits, pos) in enumerate(ordered):
        ancilla = ancilla_ids[idx % n_ancillas]
        # Comment
        dq_str = ",".join(str(q) for q in dqubits)
        lines.append(f"# Stabilizer {idx+1}/{len(ordered)}: {stype}{{{dq_str}}} ancilla={ancilla}")
        block = generate_stabilizer_block(stype, dqubits, ancilla)
        lines.extend(block)
        stab_info.append((stype, dqubits, ancilla, pos))

    return "\n".join(lines), ordered, ancilla_ids


def write_circuit_file(n_ancillas):
    """回路ファイルを生成して書き出す"""
    circuit_str, ordered, ancilla_ids = generate_circuit(n_ancillas)

    n_data = N_DATA
    all_qubits = n_data + len(ancilla_ids)

    # Stabilizer order description
    stab_descs = []
    for stype, dqubits, pos in ordered:
        dq_str = ",".join(str(q) for q in dqubits)
        stab_descs.append(f"{stype}{{{dq_str}}}")
    order_str = " -> ".join(stab_descs)

    module_content = f'''"""
d5_reuse{n_ancillas}
Distance-5 Rotated Surface Code ({n_ancillas}アンシラ再利用)

Data qubits: 0-24 (5x5 grid)
   0  1  2  3  4
   5  6  7  8  9
  10 11 12 13 14
  15 16 17 18 19
  20 21 22 23 24

Ancilla qubits: {ancilla_ids}
Total stabilizers: {len(ordered)} (12 X-type + 12 Z-type)

Hook error avoiding CNOT order:
  X stabilizers (Z-pattern): NW -> NE -> SW -> SE
  Z stabilizers (reverse-N): NW -> SW -> NE -> SE

Execution order (snake):
  {order_str}
"""

# Metadata
QUBIT_COUNT = {all_qubits}
DATA_QUBITS = {list(range(n_data))}
ANCILLA_QUBITS = {ancilla_ids}

CIRCUIT = """
{circuit_str}
"""
'''
    filename = f"d5_reuse{n_ancillas}.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(module_content)
    print(f"Generated: {filename} ({len(ordered)} stabilizers, {all_qubits} qubits)")

    # Summary
    x_count = sum(1 for s in ordered if s[0] == "X")
    z_count = sum(1 for s in ordered if s[0] == "Z")
    print(f"  X-type: {x_count}, Z-type: {z_count}")
    print(f"  Data: {n_data}, Ancilla: {ancilla_ids}")


if __name__ == "__main__":
    # Verify stabilizer count
    stabilizers = get_stabilizers()
    print(f"Total stabilizers: {len(stabilizers)}")
    x = sum(1 for s in stabilizers if s[0] == "X")
    z = sum(1 for s in stabilizers if s[0] == "Z")
    print(f"  X-type: {x}, Z-type: {z}")

    # Check all data qubits are covered
    all_data = set()
    for _, dq, _ in stabilizers:
        all_data.update(dq)
    print(f"  Data qubits covered: {sorted(all_data)}")
    print(f"  Expected: {list(range(N_DATA))}")

    ordered = get_snake_order(stabilizers)
    print(f"\nSnake order ({len(ordered)} stabilizers):")
    for i, (stype, dqubits, pos) in enumerate(ordered):
        dq_str = ",".join(str(q) for q in dqubits)
        print(f"  {i+1:2d}. {stype}{{{dq_str}}} at {pos}")

    print()
    write_circuit_file(1)
    write_circuit_file(2)

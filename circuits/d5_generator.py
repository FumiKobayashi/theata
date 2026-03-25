"""
Rotated Surface Code 回路生成器 (任意の符号距離)

Hook error avoiding CNOT order:
  X stabilizers (Z-pattern): NW -> NE -> SW -> SE
  Z stabilizers (reverse-N): NE -> SE -> NW -> SW

使い方:
  python d5_generator.py           # d=5, d=7 の reuse1/reuse2 を生成
  python d5_generator.py 5         # d=5 のみ
  python d5_generator.py 7         # d=7 のみ
"""

import sys


def qubit_id(r, c, d):
    return r * d + c


# ==========================================
# Stabilizer Definitions
# ==========================================

def get_stabilizers(d):
    """d×d rotated surface code の全スタビライザを返す"""
    stabilizers = []

    # Bulk plaquettes ((d-1)×(d-1) grid)
    for pr in range(d - 1):
        for pc in range(d - 1):
            nw = qubit_id(pr, pc, d)
            ne = qubit_id(pr, pc + 1, d)
            sw = qubit_id(pr + 1, pc, d)
            se = qubit_id(pr + 1, pc + 1, d)

            if (pr + pc) % 2 == 0:
                # X-type (Z-pattern): NW -> NE -> SW -> SE
                stabilizers.append(("X", [nw, ne, sw, se], (pr, pc)))
            else:
                # Z-type (reverse-N): NE -> SE -> NW -> SW
                stabilizers.append(("Z", [ne, se, nw, sw], (pr, pc)))

    # Boundary stabilizers
    # Left boundary (Z-type, weight 2)
    for pr in range(d - 1):
        if (pr + (-1)) % 2 == 1:  # Z-type at odd parity
            n = qubit_id(pr, 0, d)
            s = qubit_id(pr + 1, 0, d)
            # NE->SE->NW->SW: only W column exists, so NW->SW = N->S
            stabilizers.append(("Z", [n, s], ("left", pr)))

    # Right boundary (Z-type, weight 2)
    for pr in range(d - 1):
        if (pr + (d - 1)) % 2 == 1:  # Z-type at odd parity
            n = qubit_id(pr, d - 1, d)
            s = qubit_id(pr + 1, d - 1, d)
            # NE->SE->NW->SW: only E column exists, so NE->SE = N->S
            stabilizers.append(("Z", [n, s], ("right", pr)))

    # Top boundary (X-type, weight 2)
    for pc in range(d - 1):
        if ((-1) + pc) % 2 == 0:  # X-type at even parity
            w = qubit_id(0, pc, d)
            e = qubit_id(0, pc + 1, d)
            # NW->NE->SW->SE: only N row exists, so NW->NE = W->E
            stabilizers.append(("X", [w, e], ("top", pc)))

    # Bottom boundary (X-type, weight 2)
    for pc in range(d - 1):
        if ((d - 1) + pc) % 2 == 0:  # X-type at even parity
            w = qubit_id(d - 1, pc, d)
            e = qubit_id(d - 1, pc + 1, d)
            # NW->NE->SW->SE: only S row exists, so SW->SE = W->E
            stabilizers.append(("X", [w, e], ("bottom", pc)))

    return stabilizers


def get_snake_order(stabilizers, d):
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
    for pr in range(d - 1):
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
            for tb in sorted(top_boundary, key=lambda s: s[2][1]):
                ordered.append(tb)
        if pr == d - 2:
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


def generate_circuit(d, n_ancillas=1):
    """完全な回路文字列を生成

    Args:
        d: 符号距離
        n_ancillas: 使用するアンシラ数 (1 or 2)
    """
    n_data = d * d
    stabilizers = get_stabilizers(d)
    ordered = get_snake_order(stabilizers, d)

    ancilla_ids = list(range(n_data, n_data + n_ancillas))

    lines = []
    lines.append("# ============================================================")
    lines.append(f"# Distance-{d} Rotated Surface Code (reuse{n_ancillas})")
    lines.append("# ============================================================")
    lines.append("")

    # Qubit declarations
    lines.append("# Qubit Declarations")
    for i in range(n_data):
        r, c = divmod(i, d)
        lines.append(f"QUBIT {i} AT ({r}, {c})")
    for i, aid in enumerate(ancilla_ids):
        lines.append(f"QUBIT {aid} AT (10, {i})")
    lines.append("")

    # Instructions
    lines.append("# Instructions")
    for idx, (stype, dqubits, pos) in enumerate(ordered):
        ancilla = ancilla_ids[idx % n_ancillas]
        dq_str = ",".join(str(q) for q in dqubits)
        lines.append(f"# Stabilizer {idx+1}/{len(ordered)}: {stype}{{{dq_str}}} ancilla={ancilla}")
        block = generate_stabilizer_block(stype, dqubits, ancilla)
        lines.extend(block)

    return "\n".join(lines), ordered, ancilla_ids


def _grid_str(d):
    """d×d grid の qubit ID 表示文字列"""
    lines = []
    width = len(str(d * d - 1))
    for r in range(d):
        row = [str(qubit_id(r, c, d)).rjust(width) for c in range(d)]
        lines.append("  " + " ".join(row))
    return "\n".join(lines)


def write_circuit_file(d, n_ancillas):
    """回路ファイルを生成して書き出す"""
    n_data = d * d
    circuit_str, ordered, ancilla_ids = generate_circuit(d, n_ancillas)

    all_qubits = n_data + len(ancilla_ids)
    x_count = sum(1 for s in ordered if s[0] == "X")
    z_count = sum(1 for s in ordered if s[0] == "Z")

    # Stabilizer order description
    stab_descs = []
    for stype, dqubits, pos in ordered:
        dq_str = ",".join(str(q) for q in dqubits)
        stab_descs.append(f"{stype}{{{dq_str}}}")
    order_str = " -> ".join(stab_descs)

    grid = _grid_str(d)

    module_content = f'''"""
d{d}_reuse{n_ancillas}
Distance-{d} Rotated Surface Code ({n_ancillas}アンシラ再利用)

Data qubits: 0-{n_data - 1} ({d}x{d} grid)
{grid}

Ancilla qubits: {ancilla_ids}
Total stabilizers: {len(ordered)} ({x_count} X-type + {z_count} Z-type)

Hook error avoiding CNOT order:
  X stabilizers (Z-pattern): NW -> NE -> SW -> SE
  Z stabilizers (reverse-N): NE -> SE -> NW -> SW

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
    filename = f"d{d}_reuse{n_ancillas}.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(module_content)
    print(f"Generated: {filename} ({len(ordered)} stabilizers, {all_qubits} qubits)")
    print(f"  X-type: {x_count}, Z-type: {z_count}")
    print(f"  Data: {n_data}, Ancilla: {ancilla_ids}")


if __name__ == "__main__":
    distances = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [5, 7]

    for d in distances:
        n_data = d * d
        stabilizers = get_stabilizers(d)
        print(f"=== d={d} ===")
        print(f"Total stabilizers: {len(stabilizers)}")
        x = sum(1 for s in stabilizers if s[0] == "X")
        z = sum(1 for s in stabilizers if s[0] == "Z")
        print(f"  X-type: {x}, Z-type: {z}")

        # Check all data qubits are covered
        all_data = set()
        for _, dq, _ in stabilizers:
            all_data.update(dq)
        expected = set(range(n_data))
        if all_data == expected:
            print(f"  All {n_data} data qubits covered OK")
        else:
            print(f"  WARNING: missing={expected - all_data}, extra={all_data - expected}")

        ordered = get_snake_order(stabilizers, d)
        print(f"\nSnake order ({len(ordered)} stabilizers):")
        for i, (stype, dqubits, pos) in enumerate(ordered):
            dq_str = ",".join(str(q) for q in dqubits)
            print(f"  {i+1:2d}. {stype}{{{dq_str}}} at {pos}")

        print()
        write_circuit_file(d, 1)
        write_circuit_file(d, 2)
        print()

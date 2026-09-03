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


def _flip_z_cnot_order(stab, d):
    """Z スタビライザの CNOT 順序を NW->SW->NE->SE に変更する

    元: NE->SE->NW->SW (右→左走査に適した順序)
    変更後: NW->SW->NE->SE (左→右走査に適した順序)
    """
    stype, dqubits, pos = stab
    if stype != "Z" or len(dqubits) != 4:
        return stab
    # 元の順序: [NE, SE, NW, SW] → 変更後: [NW, SW, NE, SE]
    ne, se, nw, sw = dqubits
    return (stype, [nw, sw, ne, se], pos)


def _find_adjacent_top_boundary(top_boundaries, bulk_stab, d):
    """bulk Z スタビライザに隣接する top boundary X を探す"""
    if bulk_stab[0] != "Z":
        return None
    _, bulk_dq, pos = bulk_stab
    if not isinstance(pos, tuple) or not isinstance(pos[0], int):
        return None
    pr, pc = pos
    if pr != 0:  # top boundary は row 0 の bulk にのみ隣接
        return None
    # top boundary X は column pc または pc+1 付近にある
    for tb in top_boundaries:
        tb_dq = set(tb[1])
        if tb_dq & set(bulk_dq):  # 共有 qubit があれば隣接
            return tb
    return None


def _find_adjacent_bottom_boundary(bottom_boundaries, bulk_stab, d):
    """bulk Z スタビライザに隣接する bottom boundary X を探す"""
    if bulk_stab[0] != "Z":
        return None
    _, bulk_dq, pos = bulk_stab
    if not isinstance(pos, tuple) or not isinstance(pos[0], int):
        return None
    pr, pc = pos
    if pr != d - 2:  # bottom boundary は最終行の bulk にのみ隣接
        return None
    for bb in bottom_boundaries:
        bb_dq = set(bb[1])
        if bb_dq & set(bulk_dq):
            return bb
    return None


def get_snake_order(stabilizers, d):
    """蛇行順でスタビライザを並べ替える

    改善点:
    1. Z スタビライザの CNOT 順序を走査方向で切り替え
       - 左→右走査 (偶数行): NW->SW->NE->SE
       - 右→左走査 (奇数行): NE->SE->NW->SW (デフォルト)
    2. 重み2 X boundary を隣接する Z スタビライザの直後に配置
       (共有 qubit の LIZ 再利用を促進)
    """
    bulk = [s for s in stabilizers if isinstance(s[2], tuple) and isinstance(s[2][0], int)]
    left_boundary = [s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "left"]
    right_boundary = [s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "right"]
    top_boundary = list(s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "top")
    bottom_boundary = list(s for s in stabilizers if isinstance(s[2], tuple) and s[2][0] == "bottom")

    # Bulk を行ごとにグループ化
    bulk_by_row = {}
    for s in bulk:
        pr, pc = s[2]
        bulk_by_row.setdefault(pr, []).append((pc, s))

    used_top = set()
    used_bottom = set()

    # 蛇行順
    ordered = []
    for pr in range(d - 1):
        row = sorted(bulk_by_row.get(pr, []), key=lambda x: x[0])
        scan_left_to_right = (pr % 2 == 0)
        if not scan_left_to_right:
            row = list(reversed(row))

        # 行の先頭に左/右 boundary を挿入
        if scan_left_to_right:
            for lb in left_boundary:
                if lb[2][1] == pr:
                    ordered.append(lb)
        else:
            for rb in right_boundary:
                if rb[2][1] == pr:
                    ordered.append(rb)

        # Bulk スタビライザ（+ 隣接 boundary の挿入）
        for _, s in row:
            # Z スタビライザの CNOT 順序を走査方向で切り替え
            if s[0] == "Z" and scan_left_to_right:
                s = _flip_z_cnot_order(s, d)

            ordered.append(s)

            # 重み2 X boundary を隣接 Z の直後に挿入
            adj_top = _find_adjacent_top_boundary(
                [tb for tb in top_boundary if id(tb) not in used_top], s, d)
            if adj_top is not None:
                ordered.append(adj_top)
                used_top.add(id(adj_top))

            adj_bottom = _find_adjacent_bottom_boundary(
                [bb for bb in bottom_boundary if id(bb) not in used_bottom], s, d)
            if adj_bottom is not None:
                ordered.append(adj_bottom)
                used_bottom.add(id(adj_bottom))

        # 行の末尾に右/左 boundary を挿入
        if scan_left_to_right:
            for rb in right_boundary:
                if rb[2][1] == pr:
                    ordered.append(rb)
        else:
            for lb in left_boundary:
                if lb[2][1] == pr:
                    ordered.append(lb)

    # 未配置の top/bottom boundary（隣接Zがなかった場合のフォールバック）
    for tb in top_boundary:
        if id(tb) not in used_top:
            ordered.append(tb)
    for bb in bottom_boundary:
        if id(bb) not in used_bottom:
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


def stabilizer_cnot_orders(d):
    """support (frozenset) -> (type, hook-error 回避の CNOT 順 data list)。

    表面符号の各生成元について、`get_stabilizers` が定める既定の CNOT 順
    (X: NW->NE->SW->SE, Z: NE->SE->NW->SW) を support から引くための辞書。
    """
    return {frozenset(dq): (stype, list(dq)) for stype, dq, _ in get_stabilizers(d)}


def resolve_cnot_order(d, ordered, cnot_order="surface"):
    """順序付きスタビライザ列の各要素に CNOT 順を与えて正規化する。

    Args:
        ordered: [(stype, data_qubits[, pos]), ...] スタビライザの測定順序。
            外部 (配置最適化ライブラリ等) から受け取る想定。
        cnot_order:
            "surface"  = data_qubits を support として扱い、表面符号の
                         hook-error 回避順 (`stabilizer_cnot_orders`) で埋める。
                         表面符号のみを対象にする場合はこれで十分。
            "explicit" = data_qubits の並びをそのまま CNOT 順として使う。
                         一般符号に拡張する場合は呼び出し側が X/Z ごとの
                         CNOT 順を明示して渡す。

    Returns:
        [(stype, cnot_ordered_data_qubits, pos), ...] (pos が無ければ None)
    """
    if cnot_order not in ("surface", "explicit"):
        raise ValueError(f"cnot_order must be 'surface' or 'explicit', got {cnot_order!r}")
    table = stabilizer_cnot_orders(d) if cnot_order == "surface" else None
    resolved = []
    for item in ordered:
        if len(item) == 2:
            stype, dqubits = item
            pos = None
        else:
            stype, dqubits, pos = item[0], item[1], item[2]
        if cnot_order == "surface":
            key = frozenset(int(q) for q in dqubits)
            if key not in table:
                raise ValueError(
                    f"support {sorted(key)} is not a stabilizer of the d={d} "
                    f"rotated surface code")
            t_ref, cnot_list = table[key]
            if t_ref != stype:
                raise ValueError(
                    f"support {sorted(key)} is {t_ref}-type, got {stype!r}")
            resolved.append((stype, list(cnot_list), pos))
        else:
            resolved.append((stype, [int(q) for q in dqubits], pos))
    return resolved


def generate_circuit_from_order(d, ordered, n_ancillas=1, reverse=False,
                                cnot_order="surface"):
    """与えられたスタビライザ測定順序から回路文字列を生成する。

    Args:
        d: 符号距離
        ordered: [(stype, data_qubits[, pos]), ...] 測定順序 (resolve_cnot_order 参照)
        n_ancillas: 使用するアンシラ数 (1 or 2)
        reverse: True の場合、スタビライザ順を逆にし、各スタビライザ内の
            CNOT 順も逆にする (複数ラウンドの交互走査用)
        cnot_order: "surface" (support から規則で埋める) / "explicit" (与えた順)

    Returns:
        (circuit_str, ordered_resolved, ancilla_ids)
    """
    n_data = d * d
    ordered = resolve_cnot_order(d, ordered, cnot_order=cnot_order)

    if reverse:
        # スタビライザ順を逆 + 各スタビライザ内のCNOT順（data_qubits）も逆
        ordered = [(stype, list(reversed(dqubits)), pos)
                   for stype, dqubits, pos in reversed(ordered)]

    ancilla_ids = list(range(n_data, n_data + n_ancillas))

    direction = "reverse" if reverse else "forward"
    lines = []
    lines.append("# ============================================================")
    lines.append(f"# Distance-{d} Rotated Surface Code (reuse{n_ancillas}, {direction})")
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
    lines.append(f"# Instructions ({direction})")
    total = len(ordered)
    initialized_ancillas = set()
    for idx, (stype, dqubits, pos) in enumerate(ordered):
        ancilla = ancilla_ids[idx % n_ancillas]
        dq_str = ",".join(str(q) for q in dqubits)
        lines.append(f"# Stabilizer {idx+1}/{total}: {stype}{{{dq_str}}} ancilla={ancilla}")
        # 各アンシラの初回出現時に RESET を挿入
        if ancilla not in initialized_ancillas:
            lines.append(f"RESET {ancilla}")
            initialized_ancillas.add(ancilla)
        block = generate_stabilizer_block(stype, dqubits, ancilla)
        lines.extend(block)
        # MEASURE の後、同じアンシラを使うスタビライザが残っていれば RESET を挿入
        remaining = [i for i in range(idx + 1, total)
                     if ancilla_ids[i % n_ancillas] == ancilla]
        if remaining:
            lines.append(f"RESET {ancilla}")

    return "\n".join(lines), ordered, ancilla_ids


def generate_circuit(d, n_ancillas=1, reverse=False):
    """蛇行順 (get_snake_order) で完全な回路文字列を生成 (従来 API)。

    蛇行順は走査方向に応じて Z の CNOT 順を切り替えた明示的な順序を持つため
    cnot_order="explicit" で generate_circuit_from_order に渡す。
    """
    ordered = get_snake_order(get_stabilizers(d), d)
    return generate_circuit_from_order(d, ordered, n_ancillas=n_ancillas,
                                       reverse=reverse, cnot_order="explicit")


def make_circuit_module(name, d, circuit_str, ancilla_ids, ordered=None,
                        register=True):
    """回路文字列から d5_reuse1.py 相当のインメモリモジュールを作る。

    既存パイプライン (exp_cooling_pl / exp_distance_scan 等) は回路を
    `__import__(name)` で読むため、sys.modules に登録すればファイルを
    書き出さずに同じ経路でコンパイルできる。
    """
    import sys as _sys
    import types as _types
    mod = _types.ModuleType(name)
    mod.QUBIT_COUNT = d * d + len(ancilla_ids)
    mod.DATA_QUBITS = list(range(d * d))
    mod.ANCILLA_QUBITS = list(ancilla_ids)
    mod.CIRCUIT = circuit_str
    mod.ORDERED = ordered
    if register:
        _sys.modules[name] = mod
    return mod


def _grid_str(d):
    """d×d grid の qubit ID 表示文字列"""
    lines = []
    width = len(str(d * d - 1))
    for r in range(d):
        row = [str(qubit_id(r, c, d)).rjust(width) for c in range(d)]
        lines.append("  " + " ".join(row))
    return "\n".join(lines)


def write_circuit_file(d, n_ancillas, reverse=False):
    """回路ファイルを生成して書き出す"""
    n_data = d * d
    circuit_str, ordered, ancilla_ids = generate_circuit(d, n_ancillas, reverse=reverse)

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

    direction = "reverse" if reverse else "forward"
    rev_suffix = "_reverse_cnot" if reverse else ""

    module_content = f'''"""
d{d}_reuse{n_ancillas}{rev_suffix}
Distance-{d} Rotated Surface Code ({n_ancillas}アンシラ再利用, {direction})

Data qubits: 0-{n_data - 1} ({d}x{d} grid)
{grid}

Ancilla qubits: {ancilla_ids}
Total stabilizers: {len(ordered)} ({x_count} X-type + {z_count} Z-type)
Direction: {direction}
{"Reverse: stabilizer order reversed + CNOT order within each stabilizer reversed" if reverse else "Forward: standard snake scan order"}

Execution order:
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
    filename = f"d{d}_reuse{n_ancillas}{rev_suffix}.py"
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
        write_circuit_file(d, 1, reverse=True)
        write_circuit_file(d, 2, reverse=True)
        print()

"""Spatial partitioning for multi-LIZ compilation.

Rewrites the input circuit so that each stabilizer is measured by the ancilla
whose assigned LIZ is "spatially nearest" to the stabilizer's data qubits.
This reduces inter-LIZ ion transport (and therefore LIZ-passthrough congestion).

Approach:
  1. Identify stabilizer boundaries (RESET → ... → MEASURE blocks per ancilla)
  2. For each stabilizer, compute a spatial center from its data qubits
  3. Assign the stabilizer to a LIZ partition based on the center
  4. Rewrite the operations: replace the original ancilla with the LIZ's ancilla

Caveat (5/4 noted by author): this approach exploits the human-friendly
topology of rotated surface code (data qubits laid out on a 2D grid).
Generalization to other code families is future work.
"""

import dataclasses
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

from ast_def import (
    Program, Operation, GateOp, ResetOp, MeasureOp,
    ShuttleOp, SwapOp, SplitOp, MergeOp, BarrierOp,
)


@dataclass
class StabilizerInfo:
    """A single stabilizer measurement block in the program."""
    op_indices: List[int]          # indices into program.instructions
    ancilla_original: int          # ancilla used in the input circuit
    data_qubits: Set[int]          # data qubit IDs touched by MS gates


@dataclass
class SpatialPartitionResult:
    """Output of spatial partitioning."""
    new_program: Program                       # program with rewritten ancillae
    ancilla_assignments: Dict[int, int]        # liz_zone_id -> ancilla qubit ID (post-rewrite)
    stabilizer_to_liz: Dict[int, int]          # stabilizer index -> liz_zone_id
    data_to_liz_set: Dict[int, Set[int]]       # data qubit ID -> set of liz_zone_ids that use it
    pure_data: Dict[int, int]                  # data qubit ID -> the single liz_zone_id
    shared_data: Set[int]                      # data qubit IDs used by >1 LIZ


def _identify_stabilizers(
    program: Program,
    ancilla_qubits: Set[int],
) -> List[StabilizerInfo]:
    """Walk the program and identify RESET → ... → MEASURE blocks per ancilla."""
    stabilizers: List[StabilizerInfo] = []
    active: Dict[int, StabilizerInfo] = {}  # ancilla -> current open stabilizer

    for idx, op in enumerate(program.instructions):
        if isinstance(op, ResetOp):
            for t in op.targets:
                if t in ancilla_qubits:
                    # New stabilizer for this ancilla
                    active[t] = StabilizerInfo(
                        op_indices=[idx],
                        ancilla_original=t,
                        data_qubits=set(),
                    )
            continue

        if isinstance(op, MeasureOp):
            for t in op.targets:
                if t in ancilla_qubits and t in active:
                    active[t].op_indices.append(idx)
                    stabilizers.append(active.pop(t))
            continue

        # MS/R/Other: attribute to the active stabilizer whose ancilla is involved
        owners = []
        if isinstance(op, GateOp):
            owners = [t for t in op.targets if t in active]
        else:
            owners = [t for t in getattr(op, "targets", []) if t in active]

        if not owners:
            # Op not part of any open stabilizer (shouldn't happen for clean input)
            continue

        owner = owners[0]
        active[owner].op_indices.append(idx)
        if isinstance(op, GateOp) and op.type == "MS":
            for t in op.targets:
                if t != owner:
                    active[owner].data_qubits.add(t)

    return stabilizers


def _classify_stabilizer_row(
    stab: StabilizerInfo,
    qubit_row: Dict[int, float],
    code_distance: int,
    num_partitions: int,
) -> int:
    """Return partition index (0..num_partitions-1) based on average row of data qubits.

    Row range [0, d-1] is split evenly into num_partitions bands.
    """
    if not stab.data_qubits:
        return 0
    avg_row = sum(qubit_row[q] for q in stab.data_qubits) / len(stab.data_qubits)
    band = int(avg_row * num_partitions / code_distance)
    return min(band, num_partitions - 1)


def _replace_targets(op: Operation, mapping: Dict[int, int]) -> Operation:
    """Return a new op with targets remapped via `mapping`."""
    if not hasattr(op, "targets"):
        return op
    new_targets = [mapping.get(t, t) for t in op.targets]
    if new_targets == op.targets:
        return op
    return dataclasses.replace(op, targets=new_targets)


def partition_spatial_row(
    program: Program,
    ancilla_qubits: List[int],
    liz_zone_ids: List[int],
    code_distance: int,
    phase_alternation: bool = False,
) -> SpatialPartitionResult:
    """Build a row-band spatial partition.

    Args:
        program: parsed input program
        ancilla_qubits: list of ancilla qubit IDs in the program (e.g., [25, 26] for d5_reuse2)
        liz_zone_ids: list of LIZ zone IDs, ordered by physical position (left to right)
        code_distance: side length of the rotated surface code grid
        phase_alternation: if True, reorder each LIZ's stabilizer chain so that
            LIZs with even index do pure stabilizers first then shared,
            while LIZs with odd index do shared first then pure.
            This anti-correlates central-memory accesses across the two LIZs:
            while LIZ-A processes its 7 pure stabilizers (no central-M traffic),
            LIZ-B processes its 5 shared stabilizers (using central-M), and
            vice versa.

    Stabilizers are split into len(liz_zone_ids) horizontal bands by data-qubit
    row average. Within each band, the stabilizer is reassigned to the
    band-corresponding ancilla (and therefore the corresponding LIZ).
    """
    if len(ancilla_qubits) < len(liz_zone_ids):
        raise ValueError(
            f"need at least {len(liz_zone_ids)} ancillae for {len(liz_zone_ids)} LIZs, "
            f"got {len(ancilla_qubits)}"
        )

    # Build row lookup from declarations: location.trap == row in our convention
    qubit_row: Dict[int, float] = {}
    for decl in program.declarations:
        qubit_row[decl.id] = float(decl.location.trap)

    ancilla_set = set(ancilla_qubits)
    stabilizers = _identify_stabilizers(program, ancilla_set)

    # Pick one ancilla per LIZ (use input order; remaining ancillae are unused
    # — caller can prune them or we just rewrite all stabs to use these).
    liz_ancilla = {
        liz_zone_ids[i]: ancilla_qubits[i] for i in range(len(liz_zone_ids))
    }

    # Classify each stabilizer
    stab_to_liz: Dict[int, int] = {}
    rewrite_per_stab: List[Dict[int, int]] = []  # per-stabilizer ancilla rewrite map
    for s_idx, stab in enumerate(stabilizers):
        band = _classify_stabilizer_row(stab, qubit_row, code_distance, len(liz_zone_ids))
        target_liz = liz_zone_ids[band]
        target_ancilla = liz_ancilla[target_liz]
        stab_to_liz[s_idx] = target_liz
        rewrite_per_stab.append({stab.ancilla_original: target_ancilla})

    # Build data-qubit usage info (needed for phase classification)
    data_to_liz: Dict[int, Set[int]] = {}
    for s_idx, stab in enumerate(stabilizers):
        liz = stab_to_liz[s_idx]
        for d in stab.data_qubits:
            data_to_liz.setdefault(d, set()).add(liz)
    pure_data = {d: next(iter(s)) for d, s in data_to_liz.items() if len(s) == 1}
    shared_data = {d for d, s in data_to_liz.items() if len(s) > 1}

    # Determine emission order per LIZ chain
    # liz_to_stab_order[liz_id] = ordered list of stabilizer indices to emit
    liz_to_stab_order: Dict[int, List[int]] = {liz: [] for liz in liz_zone_ids}
    for s_idx in range(len(stabilizers)):
        liz_to_stab_order[stab_to_liz[s_idx]].append(s_idx)

    if phase_alternation:
        for liz_idx, liz in enumerate(liz_zone_ids):
            indices = liz_to_stab_order[liz]
            pure_idx = [
                i for i in indices
                if not any(d in shared_data for d in stabilizers[i].data_qubits)
            ]
            shared_idx = [
                i for i in indices
                if any(d in shared_data for d in stabilizers[i].data_qubits)
            ]
            if liz_idx % 2 == 0:
                liz_to_stab_order[liz] = pure_idx + shared_idx
            else:
                liz_to_stab_order[liz] = shared_idx + pure_idx

    # Emit new instruction list: chain by chain (stabilizer_parallel makes
    # cross-chain order irrelevant for DAG dependencies)
    if phase_alternation:
        new_instructions: List[Operation] = []
        for liz in liz_zone_ids:
            for s_idx in liz_to_stab_order[liz]:
                stab = stabilizers[s_idx]
                mapping = rewrite_per_stab[s_idx]
                for op_idx in stab.op_indices:
                    new_instructions.append(
                        _replace_targets(program.instructions[op_idx], mapping)
                    )
    else:
        # Preserve input ordering, only rewrite ancillae
        new_instructions = list(program.instructions)
        for s_idx, stab in enumerate(stabilizers):
            mapping = rewrite_per_stab[s_idx]
            if not mapping:
                continue
            for op_idx in stab.op_indices:
                new_instructions[op_idx] = _replace_targets(
                    new_instructions[op_idx], mapping
                )

    new_program = Program(
        declarations=list(program.declarations),
        instructions=new_instructions,
    )

    return SpatialPartitionResult(
        new_program=new_program,
        ancilla_assignments=liz_ancilla,
        stabilizer_to_liz=stab_to_liz,
        data_to_liz_set=data_to_liz,
        pure_data=pure_data,
        shared_data=shared_data,
    )


def compute_initial_placement_3tier(
    result: SpatialPartitionResult,
    architecture,
    all_data_qubits: List[int],
) -> Dict[int, int]:
    """Compute a 3-tier initial placement for 2-LIZ architecture.

    Pure-A data → memory zone left of LIZ-A
    Pure-B data → memory zone right of LIZ-B
    Shared data → memory zone between the two LIZs
    Each ancilla → memory zone immediately left of its LIZ

    Generalization to >2 LIZs: each pure-i data goes to the memory zone
    immediately left of LIZ-i; shared data goes to the central-most memory.
    """
    memory_zones = [z.id for z in architecture.zones if z.zone_type == "M"]
    liz_zones = [z.id for z in architecture.zones if z.zone_type == "L"]
    num_liz = len(liz_zones)

    # Map each LIZ to the memory zone immediately to its left
    # For T-M-T-L-T-M-T-L-...-T-M-T, left-of-LIZ_i is memory_zones[i]
    liz_left_memory = {liz_zones[i]: memory_zones[i] for i in range(num_liz)}
    # Right-most memory (only relevant for the right boundary)
    rightmost_memory = memory_zones[-1]
    # Central memory: between the two LIZs (for 2-LIZ case)
    central_memory = memory_zones[num_liz - 1] if num_liz >= 2 else memory_zones[0]

    placement: Dict[int, int] = {}

    # Ancillae: place each in the memory immediately left of its LIZ
    for liz_id, ancilla_id in result.ancilla_assignments.items():
        placement[ancilla_id] = liz_left_memory[liz_id]

    # Data qubits: distribute by classification
    for d in all_data_qubits:
        if d in result.pure_data:
            liz_id = result.pure_data[d]
            # Pure data goes to the LIZ's left memory (or right memory for the rightmost LIZ)
            if liz_id == liz_zones[-1]:
                # rightmost LIZ: pure data goes right of LIZ
                placement[d] = rightmost_memory
            else:
                placement[d] = liz_left_memory[liz_id]
        elif d in result.shared_data:
            placement[d] = central_memory
        else:
            # Data qubit not touched by any stabilizer (shouldn't happen for full circuit)
            placement[d] = memory_zones[0]

    return placement

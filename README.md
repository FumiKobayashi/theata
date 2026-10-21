# Transport-Aware Syndrome Measurement Circuit Compiler for 1D QCCD Architecture

A compiler and simulator for syndrome measurement circuits on a one-dimensional QCCD (quantum charge-coupled device) architecture.

## Project Layout

```
src/                  Compiler core
circuits/             Input circuit definitions
experiments/          Experiment scripts
results/              Experiment outputs
plots/                Plot outputs
ThesisImages/         Paper and poster assets
```

## Using the Compiler

### Basic Workflow

```python
from qccd_parser import parse
from dag_builder import DAGBuilder
from qccd_compiler import QCCDArchitecture, QCCDCompiler, CompilerConfig
from parallel_estimator import ParallelEstimator
from validator import validate_operations

# 1. Parse the circuit and build the DAG
program = parse(circuit_module.CIRCUIT)
dag = DAGBuilder(program, stabilizer_parallel=False).build()

# 2. Architecture and initial placement
arch = QCCDArchitecture.simple_mlm(liz_capacity=4, memory_capacity=20, temp_capacity=20)
initial = {0: 1, 1: 1, 2: 5, ...}  # ion_id -> zone_id

# 3. Compile
config = CompilerConfig(eviction_strategy="balanced")
compiler = QCCDCompiler(arch, config=config)
result = compiler.compile(dag, initial)

# 4. Use the result (see below)
```

### Structure of CompileResult

`compiler.compile()` returns a `CompileResult` with the following fields:

| Field | Type | Description |
|-----------|-----|------|
| `timesteps` | `List[TimeStep]` | Parallelized instruction sequence (Scheduler output) |
| `flat_ops` | `List[Operation]` | Sequential instruction sequence before parallelization (MovementPlanner output) |
| `operations` | `List[Operation]` | Flattened view of `timesteps` (backward-compatibility property) |
| `stats` | `Dict[str, int]` | Counts per operation type (gates, splits, merges, ...) |
| `phonon_data` | `Dict` | Phonon-tracking results |
| `final_placement` | `Dict[int, int]` | Ion placement at the end of the round (ion_id -> zone_id) |
| `final_zone_contents` | `Dict[int, List[int]]` | Ion ordering inside each zone at the end of the round |
| `config` | `CompilerConfig` | Compiler configuration that was used |

### Working with CompileResult

#### Estimating Execution Time

```python
from parallel_estimator import ParallelEstimator

est = ParallelEstimator()
r = est.estimate(result.timesteps)
print(f"Parallel time: {r.total_time / 1000:.2f} ms")
print(f"Sequential time: {r.sequential_time / 1000:.2f} ms")
print(f"Speedup: {r.speedup:.2f}x")
```

#### Validating the Instruction Sequence

```python
from validator import validate_operations, validate_timesteps

# Sequential validation
seq_result = validate_operations(result.operations, initial, arch)

# Parallel validation (includes zone-conflict and ion-conflict checks)
par_result = validate_timesteps(result.timesteps, initial, arch)
```

#### Text Output (Human-Readable Instruction Sequence)

```python
from parallel_estimator import save_parallel_txt

save_parallel_txt(result.timesteps, "output.txt",
                  initial_placement=initial, arch=arch)
```

Output format (one operation per line; parallel operations expanded as a block):
```
TS   0 [  100.00μs  @    100.00μs]  Split(ion8) z1->z2
TS   1 [  100.00μs  @    200.00μs]  [2P]
  | Split(ion9) z5->z4                    100.00μs
  | Swap(0,1) z1                           20.00μs
```

#### Reading Phonon Numbers

```python
pd = result.phonon_data
print(f"Max axial phonon: {pd['max_axial']:.2f}")
print(f"Max transversal phonon: {pd['max_transversal']:.2f}")

# Phonon numbers at each MS gate
for gate_info in pd['at_gates']:
    print(f"{gate_info['gate']}: {gate_info['phonons']}")
```

#### Carrying the Placement Across Rounds

```python
# Round 1
result1 = compiler.compile(dag1, initial)

# Round 2: preserve the ion ordering of final_zone_contents
ordered_placement = {}
for zone_id in sorted(result1.final_zone_contents.keys()):
    for ion_id in result1.final_zone_contents[zone_id]:
        ordered_placement[ion_id] = zone_id

result2 = compiler.compile(dag2, ordered_placement)
```

### CompilerConfig Options

| Option | Type | Default | Description |
|-----------|-----|-----------|------|
| `gate_selection` | str | `"min_id"` | Gate selection strategy |
| `memory_reorder` | bool | `False` | Lookahead reordering inside Memory during Measure |
| `measure_batch_size` | int | `1` | Measure batch size (2+ enables batching) |
| `interleave` | bool | `False` | Interleaved stabilizer execution |
| `eviction_strategy` | str | `"balanced"` | Eviction strategy out of the LIZ (`"balanced"` or `"nearest"`) |

## Modules in src/

| File | Role |
|---------|------|
| `qccd_parser.py` | Parser for the custom DSL (built on Lark) |
| `ast_def.py` | Type definitions for operations (GateOp, SplitOp, ShuttleOp, ...) |
| `dag_builder.py` | Quantum circuit → DAG conversion, stabilizer detection, hook-error edge injection |
| `qccd_compiler.py` | MovementPlanner + Scheduler + QCCDState (compiler core) |
| `estimator.py` | Sequential execution-time estimation + TimeParameters definition |
| `parallel_estimator.py` | Parallel execution-time estimation + text output |
| `validator.py` | Constraint validation of instruction sequences (sequential + parallel) |
| `memory_reorder_optimizer.py` | Ion-reordering optimization inside Memory during Measure |

## Circuit File Format

Each `.py` file in `circuits/` carries the following metadata and circuit string:

```python
QUBIT_COUNT = 10
DATA_QUBITS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
ANCILLA_QUBITS = [9]

CIRCUIT = """
QUBIT 0 AT (0, 0)
...
RESET 9
R 0 1.5708 1.5708
MS 0 9 0.25
...
MEASURE 9
"""
```

Circuits for d=5 and d=7 can be generated with `d5_generator.py`:
```bash
cd circuits
python d5_generator.py 5 7    # generates reuse1, reuse2, and reverse_cnot for d5/d7
```

## Architecture

Seven-zone one-dimensional layout: `T(20) - M(20) - T(20) - L(4) - T(20) - M(20) - T(20)`

- T: Temporary zone (staging area during transport)
- M: Memory zone (ion storage)
- L: LIZ (laser interaction zone, where gates execute)

## Experiment Scripts

Experiment scripts live in `experiments/` and analyze the compiler output in various ways.
Each script accepts command-line options (see `--help`).

| File | Purpose | Main output directory |
|---------|------|-----------|
| `exp_parallel.py` | Parallelism analysis of a single circuit | `results/parallel/` |
| `exp_placement.py` | Comparison of initial-placement strategies | `results/placement/` |
| `exp_multi_round.py` | Multi-round QEC simulation | `results/multi_round/` |
| `exp_memory_reorder.py` | Effect of Memory reordering optimization | `results/memory_reorder/` |
| `exp_interleave.py` | Effect of interleaved stabilizer execution | `results/interleave/` |
| `exp_noise_injection.py` | Phonon-based noise injection into Stim | `results/noise_injection/` |

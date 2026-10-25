# Guarded re-run (2026-10-25) of the 10/23-24 sweeps, after the cooling-rule
# fix (no CoolOp on a chain holding a gated, unmeasured ancilla).
# Same grids and settings as run_sweeps_1024.ps1; only the output roots differ.
# ASCII-only (PowerShell 5.1 reads BOM-less UTF-8 as ANSI).
#
#   Sweep 1: theta sweep at t = 1 ms, nbar_res = 0 (theta_sweep_1025guard)
#   Sweep 2: abstract Cool (t, nbar_res) grid, theta = 1.5 x 16 (req_grid_1025guard)
#   Sweep 3: theta x other cooling parameters (theta_cross_1025guard)
# All: arm = resident-pred (res@2c + pred + anc), mainz_heating, 100k shots.
# Done points (a JSON in the dir) are skipped on re-run.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_sweeps_1025guard.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

$common = @(
    'experiments/exp_cooling_distance_scan.py',
    '--arms', 'resident-pred',
    '--distances', '3', '5', '7',
    '--pred-max-iters', '16',
    '--params', 'params/mainz_heating.json',
    '--shots', '100000',
    '--quiet-clip'
)

function Invoke-Point($outDir, $extraArgs) {
    if (Test-Path "$outDir/*.json") {
        Write-Output "skip (done): $outDir"
        return
    }
    Write-Output "=== run: $outDir ==="
    python @common @extraArgs --out-dir $outDir
}

$thetas = @('0.7', '1.4', '1.5', '2.1', '2.8', '3.5', '4.2', '5.6')

# --- Sweep 1: theta ---
foreach ($th in $thetas) {
    Invoke-Point "results/theta_sweep_1025guard/th$th" @(
        '--pred-threshold', $th,
        '--cool-us', '1000', '--coolant-recool-us', '1000',
        '--ancilla-cool-us', '1000', '--cool-residual', '0')
}

# --- Sweep 2: (t, nbar_res) grid ---
foreach ($t in 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000) {
    $tms = $t / 1000
    foreach ($r in @('0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0')) {
        Invoke-Point "results/req_grid_1025guard/t${tms}ms_r$r" @(
            '--pred-threshold', '1.5',
            '--cool-us', $t, '--coolant-recool-us', $t,
            '--ancilla-cool-us', $t, '--cool-residual', $r)
    }
}

# --- Sweep 3: theta x other cooling parameters ---
$crossCombos = @(
    @{ t = 1000; r = '0.5' },
    @{ t = 1000; r = '1.0' },
    @{ t = 1000; r = '1.75' },
    @{ t = 5000; r = '0' }
)
foreach ($combo in $crossCombos) {
    $t = $combo.t
    $tms = $t / 1000
    foreach ($th in $thetas) {
        Invoke-Point "results/theta_cross_1025guard/t${tms}ms_r$($combo.r)/th$th" @(
            '--pred-threshold', $th,
            '--cool-us', $t, '--coolant-recool-us', $t,
            '--ancilla-cool-us', $t, '--cool-residual', $combo.r)
    }
}
Write-Output "ALL GUARDED SWEEPS DONE"

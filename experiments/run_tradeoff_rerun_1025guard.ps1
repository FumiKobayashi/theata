# Re-run of the pl_vs_time_tradeoff hybrid data points after the cooling-rule
# guard (2026-10-25): no CoolOp while a gated, unmeasured ancilla shares the
# chain. Conditions replicate the originals (resident_1006 / combo_cycle_
# sweep_1020 / pred_data_1007 combo_x4_eq): anc=2000us, recool=400us, default
# cool duration and residuals, mainz_heating, 3 rounds -- now at 100k shots
# and all three distances for every point.
# ASCII-only (PowerShell 5.1 reads BOM-less UTF-8 as ANSI).
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_tradeoff_rerun_1025guard.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

$common = @(
    'experiments/exp_cooling_distance_scan.py',
    '--distances', '3', '5', '7',
    '--ancilla-cool-us', '2000',
    '--coolant-recool-us', '400',
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

Invoke-Point "results/tradeoff_rerun_1025guard/resident/main" @('--arms', 'resident')
foreach ($n in 1, 2, 4, 8, 16) {
    Invoke-Point "results/tradeoff_rerun_1025guard/x$n/main" @(
        '--arms', 'resident-pred', '--pred-threshold', '1.5',
        '--pred-max-iters', $n)
}
Write-Output "TRADEOFF RERUN DONE"

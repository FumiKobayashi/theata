# Theta sweep of the isolated-recool strategy (2026-10-25 author proposal):
# evict the entangled ancilla to memory, cool the coolants only, bring it back.
# Same conditions as theta_sweep_1025guard (t = 1 ms, nbar_res = 0, 100k shots),
# plus --isolate-recool. Done points are skipped. ASCII-only.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_isolate_theta_1025.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

foreach ($th in @('0.7', '1.4', '1.5', '2.1', '2.8', '3.5', '4.2', '5.6')) {
    $outDir = "results/isolate_recool_1025/th$th"
    if (Test-Path "$outDir/*.json") {
        Write-Output "skip (done): $outDir"
        continue
    }
    Write-Output "=== run: $outDir ==="
    python experiments/exp_cooling_distance_scan.py `
        --arms resident-pred --distances 3 5 7 `
        --pred-threshold $th --pred-max-iters 16 --isolate-recool `
        --cool-us 1000 --coolant-recool-us 1000 --ancilla-cool-us 1000 `
        --cool-residual 0 `
        --params params/mainz_heating.json --shots 100000 --quiet-clip `
        --out-dir $outDir
}
Write-Output "ISOLATE THETA SWEEP DONE"

# Iterated isolated re-cool test (2026-10-26): repeat the evict->Cool->return
# cycle k times per isolation event. Conditions match isolate_recool_1025
# (resident-pred x16, t = 1 ms, nbar_res = 0, 100k shots). k = 1 results live
# in results/isolate_recool_1025/th{theta}. ASCII-only.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_isolate_iter_1026.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

foreach ($combo in @(
    @{ th = '2.8'; k = 2 }, @{ th = '2.8'; k = 3 },
    @{ th = '4.2'; k = 2 }, @{ th = '4.2'; k = 3 })) {
    $outDir = "results/isolate_iter_1026/th$($combo.th)_k$($combo.k)"
    if (Test-Path "$outDir/*.json") {
        Write-Output "skip (done): $outDir"
        continue
    }
    Write-Output "=== run: $outDir ==="
    python experiments/exp_cooling_distance_scan.py `
        --arms resident-pred --distances 3 5 7 `
        --pred-threshold $combo.th --pred-max-iters 16 `
        --isolate-recool --isolate-recool-iters $combo.k `
        --cool-us 1000 --coolant-recool-us 1000 --ancilla-cool-us 1000 `
        --cool-residual 0 `
        --params params/mainz_heating.json --shots 100000 --quiet-clip `
        --out-dir $outDir
}
Write-Output "ISOLATE ITER TEST DONE"

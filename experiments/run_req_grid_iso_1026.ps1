# Abstract Cool (t, nbar_res) grid WITH isolated re-cool, k = 1 (2026-10-26).
# Same grid and settings as req_grid_1025guard (resident-pred, theta = 1.5 x 16,
# mainz_heating, 100k shots) plus --isolate-recool, so the pair of heatmaps
# isolates the effect of the legal in-window re-cool. Done points are skipped.
# ASCII-only.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_req_grid_iso_1026.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

foreach ($t in 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000) {
    $tms = $t / 1000
    foreach ($r in @('0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0')) {
        $outDir = "results/req_grid_1026iso/t${tms}ms_r$r"
        if (Test-Path "$outDir/*.json") {
            Write-Output "skip (done): $outDir"
            continue
        }
        Write-Output "=== run: $outDir ==="
        python experiments/exp_cooling_distance_scan.py `
            --arms resident-pred --distances 3 5 7 `
            --pred-threshold 1.5 --pred-max-iters 16 --isolate-recool `
            --cool-us $t --coolant-recool-us $t --ancilla-cool-us $t `
            --cool-residual $r `
            --params params/mainz_heating.json --shots 100000 --quiet-clip `
            --out-dir $outDir
    }
}
Write-Output "ISO GRID DONE"

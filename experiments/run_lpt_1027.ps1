# LIZ predictive trigger first measurements (2026-10-27).
# Arm = coolant-present + LIZ cooling (res@2c+anc), t = 1 ms, nbar_res = 0,
# mainz_heating, 100k shots. temp_th1.5x4 is already measured.
# ASCII-only. Done points are skipped.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_lpt_1027.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

$points = @(
    @{ dir = 'memory_th1.5x4'; th = '1.5'; iters = 4; park = 'memory' },
    @{ dir = 'temp_th0.7x4';   th = '0.7'; iters = 4; park = 'temp' },
    @{ dir = 'temp_th2.1x4';   th = '2.1'; iters = 4; park = 'temp' },
    @{ dir = 'temp_th2.8x4';   th = '2.8'; iters = 4; park = 'temp' }
)
foreach ($p in $points) {
    $outDir = "results/lpt_1027/$($p.dir)"
    if (Test-Path "$outDir/*.json") {
        Write-Output "skip (done): $outDir"
        continue
    }
    Write-Output "=== run: $outDir ==="
    python experiments/exp_cooling_distance_scan.py `
        --arms resident --distances 3 5 7 `
        --liz-pt $p.th --liz-pt-iters $p.iters --liz-pt-park $p.park `
        --cool-us 1000 --coolant-recool-us 1000 --ancilla-cool-us 1000 `
        --cool-residual 0 `
        --params params/mainz_heating.json --shots 100000 --quiet-clip `
        --out-dir $outDir
}
Write-Output "LPT POINTS DONE"

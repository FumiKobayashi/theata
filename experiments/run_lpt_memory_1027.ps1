# Memory-park LPT theta sweep (2026-10-27 evening). Same conditions as the
# temp-park family (resident arm + LPT, t = 1 ms, nbar_res = 0, mainz, 100k).
# theta = 1.5 already exists (results/lpt_1027/memory_th1.5x4). ASCII-only.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_lpt_memory_1027.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

$points = @(
    @{ dir = 'memory_th0.7x4';       args = @('--liz-pt', '0.7') },
    @{ dir = 'memory_th2.1x4';       args = @('--liz-pt', '2.1') },
    @{ dir = 'memory_th2.8x4';       args = @('--liz-pt', '2.8') },
    @{ dir = 'memory_th1.5to0.75x4'; args = @('--liz-pt', '1.5', '--liz-pt-target', '0.75') }
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
        @($p.args) --liz-pt-iters 4 --liz-pt-park memory `
        --cool-us 1000 --coolant-recool-us 1000 --ancilla-cool-us 1000 `
        --cool-residual 0 `
        --params params/mainz_heating.json --shots 100000 --quiet-clip `
        --out-dir $outDir
}
Write-Output "LPT MEMORY SWEEP DONE"

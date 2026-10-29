# LPT theta-only sweep under the sufficiency convention (user decision
# 2026-10-28): N_max = 32 is "large enough" so that nearly every gate runs
# at nbar <= theta + 0.1; the cycle loop exits early once theta is reached,
# so the extra budget costs nothing. Both isolation variants, unified
# convention (t = 1 ms, nbar_res = 0, mainz, 100k shots). Resumable.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_lpt_theta_n32_1028.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

foreach ($park in @('temp', 'memory')) {
    foreach ($th in @('2.8', '2.1', '1.5', '0.7', '0.5', '0.35', '0.2')) {
        $outDir = "results/lpt_n32_1028/${park}_th$th"
        if (Test-Path "$outDir/*.json") {
            Write-Output "skip (done): $outDir"
            continue
        }
        Write-Output "=== run: $outDir ==="
        python experiments/exp_cooling_distance_scan.py `
            --arms resident --distances 3 5 7 `
            --liz-pt $th --liz-pt-iters 32 --liz-pt-park $park `
            --cool-us 1000 --coolant-recool-us 1000 --ancilla-cool-us 1000 `
            --cool-residual 0 `
            --params params/mainz_heating.json --shots 100000 --quiet-clip `
            --out-dir $outDir
    }
}
Write-Output "LPT N32 THETA SWEEP DONE"

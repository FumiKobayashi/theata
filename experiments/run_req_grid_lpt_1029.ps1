# Requirement grid (t_cool, nbar_res) for the canonical LPT strategy
# (user decision 2026-10-28: run the sweep on the best method).
# Strategy: coolant x2 + LIZ cooling + LIZ PT, temp isolation, N_max = 32
# (sufficiency convention). Per-row trigger point theta(r) = max(0.2, r + 0.1)
# so the trigger never demands less than the coolant residual allows
# (otherwise every cell with r >= theta thrashes at its floor).
# Grid: t = 1..10 ms x nbar_res = 0..2.0 step 0.25, mainz, 100k shots.
# ASCII-only, resumable.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_req_grid_lpt_1029.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

foreach ($t in 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000) {
    $tms = $t / 1000
    foreach ($r in @('0', '0.25', '0.5', '0.75', '1.0', '1.25', '1.5', '1.75', '2.0')) {
        $rv = [double]$r
        $th = [math]::Max(0.2, $rv + 0.1)
        $thStr = $th.ToString([System.Globalization.CultureInfo]::InvariantCulture)
        $outDir = "results/req_grid_lpt_1029/t${tms}ms_r$r"
        if (Test-Path "$outDir/*.json") {
            Write-Output "skip (done): $outDir"
            continue
        }
        Write-Output "=== run: $outDir (theta=$thStr) ==="
        python experiments/exp_cooling_distance_scan.py `
            --arms resident --distances 3 5 7 `
            --liz-pt $thStr --liz-pt-iters 32 --liz-pt-park temp `
            --cool-us $t --coolant-recool-us $t --ancilla-cool-us $t `
            --cool-residual $r `
            --params params/mainz_heating.json --shots 100000 --quiet-clip `
            --out-dir $outDir
    }
}
Write-Output "LPT REQ GRID DONE"

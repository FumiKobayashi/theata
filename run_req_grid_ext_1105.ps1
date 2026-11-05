# Requirement-grid extension rows (2026-11-05 meeting):
# make the failure region visible by extending the residual axis past
# the predicted boundary r ~= 2.6. Adds r = 2.5..4.0 (0.5 steps) x
# t = 1..10 ms, same configuration as results/req_grid_lpt_1029
# (LPT temp N_max=32, theta = r+0.1, 100k shots, Mainz heating).
# Resumable: cells with an existing JSON are skipped.
$ErrorActionPreference = "Stop"
$exp = "experiments/exp_cooling_distance_scan.py"
$outRoot = "results/req_grid_lpt_ext_1105"

foreach ($r in @(2.5, 3.0, 3.5, 4.0)) {
    foreach ($t in 1..10) {
        $outDir = "$outRoot/t${t}ms_r${r}"
        if (Test-Path "$outDir/*.json") {
            Write-Output "skip t=${t}ms r=$r"
            continue
        }
        $theta = $r + 0.1
        $coolUs = $t * 1000
        & python $exp --arms resident --distances 3 5 7 `
            --resident-coolants 2 --coolant-recool-us $coolUs `
            --ancilla-cool-us $coolUs `
            --cool-us $coolUs --cool-residual $r `
            --liz-pt $theta --liz-pt-iters 32 --liz-pt-park temp `
            --shots 100000 --seed 2026 --quiet-clip `
            --params params/mainz_heating.json --out-dir $outDir
        Write-Output "done t=${t}ms r=$r"
    }
}
Write-Output "req grid extension done"

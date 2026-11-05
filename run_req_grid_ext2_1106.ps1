# Requirement-grid duration extension (2026-11-06, sensei request):
# make the failure boundary visibly diagonal on log-log axes by
# extending the duration axis past 10 ms. Log-linear fits on the
# existing grid put the break-even duration at ~14-42 ms (L35) and
# ~12-72 ms (L57) depending on the residual, so t = 15/22/33/50/75 ms
# brackets both boundaries down to r = 0.
# Residual sampling matches the 1029 grid (0.25 steps to 2.0, then
# 0.5); r = 2.5/3.0 are dropped for t >= 33 ms (deep fail for both
# pairs by the fits). Same configuration as results/req_grid_lpt_1029
# (LPT temp N_max=32, theta = max(0.2, r+0.1), 100k shots, Mainz).
# Resumable: cells with an existing JSON are skipped.
$ErrorActionPreference = "Stop"
$exp = "experiments/exp_cooling_distance_scan.py"
$outRoot = "results/req_grid_lpt_ext2_1106"

$rAll = @('0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0','2.5','3.0')
$rLow = @('0','0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0')

foreach ($t in @(15, 22, 33, 50, 75)) {
    $rList = if ($t -ge 33) { $rLow } else { $rAll }
    foreach ($rs in $rList) {
        $r = [double]$rs
        $outDir = "$outRoot/t${t}ms_r${rs}"
        if (Test-Path "$outDir/*.json") {
            Write-Output "skip t=${t}ms r=$rs"
            continue
        }
        $theta = [Math]::Max(0.2, $r + 0.1)
        $coolUs = $t * 1000
        & python $exp --arms resident --distances 3 5 7 `
            --resident-coolants 2 --coolant-recool-us $coolUs `
            --ancilla-cool-us $coolUs `
            --cool-us $coolUs --cool-residual $r `
            --liz-pt $theta --liz-pt-iters 32 --liz-pt-park temp `
            --shots 100000 --seed 2026 --quiet-clip `
            --params params/mainz_heating.json --out-dir $outDir
        Write-Output "done t=${t}ms r=$rs"
    }
}
Write-Output "req grid ext2 done"

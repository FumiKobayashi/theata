# Timing probe for the d<=13 nbar-cap scan estimate (2026-11-05).
# Measures one cap point of the no-cool scan at d=7 and d=9, each at two
# shot counts, so compile cost and per-shot sampling cost separate.
$ErrorActionPreference = "Stop"
$exp = "experiments/exp_cooling_distance_scan.py"
$outRoot = "results/d13_estimate_1105"

foreach ($d in @(7, 9)) {
    foreach ($shots in @(10000, 50000)) {
        $outDir = "$outRoot/d${d}_s${shots}"
        if (Test-Path "$outDir/*.json") {
            Write-Output "skip d=$d shots=$shots"
            continue
        }
        $t = Measure-Command {
            & python $exp --arms no-cool --distances $d `
                --nbar-cap 2.0 --shots $shots --seed 2026 --quiet-clip `
                --params params/mainz_heating.json --out-dir $outDir
        }
        Write-Output ("TIMING d={0} shots={1} total_s={2:N1}" -f $d, $shots, $t.TotalSeconds)
    }
}
Write-Output "d13 estimate probe done"

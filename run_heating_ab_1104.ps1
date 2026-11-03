# Heating A/B for the Section 6 sensitivity paragraph (2026-11-04).
# Re-measures the +24% background-heating claim in the guarded world:
# merge policy at d=5, unified cooling time t=1ms, 100k shots,
# with (mainz) and without (per-op only) background heating.
# Resumable: skips a run if its output JSON already exists.
$ErrorActionPreference = "Stop"
$py = "python"
$exp = "experiments/exp_cooling_distance_scan.py"

$runs = @(
    @{ name = "noheat"; params = $null },
    @{ name = "mainz";  params = "params/mainz_heating.json" }
)
foreach ($r in $runs) {
    $outDir = "results/heating_ab_1104/$($r.name)"
    if (Test-Path "$outDir/*.json") {
        Write-Output "skip $($r.name): JSON exists"
        continue
    }
    $argList = @($exp, "--arms", "merge",
        "--distances", "5",
        "--merge-cool-threshold", "1.5",
        "--ancilla-cool-us", "1000",
        "--cool-us", "1000",
        "--shots", "100000", "--seed", "2026", "--quiet-clip",
        "--out-dir", $outDir)
    if ($r.params) { $argList += @("--params", $r.params) }
    Write-Output "=== run $($r.name) ==="
    & $py @argList
    if ($LASTEXITCODE -ne 0) { throw "run $($r.name) failed" }
}
Write-Output "heating A/B done"

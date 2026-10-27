# Five-strategy comparison data at the unified abstract-Cool convention
# (t = 1 ms for every cooling duration, nbar_res = 0, mainz, 100k shots).
# Fills the points missing for the two-panel pl_vs_time_tradeoff v5:
#   no coolant: LIZ cooling only (anc), MPT N in {1,4,16} (pred)
#   coolant:    LIZ cooling only (resident), MPT N in {1,4} (resident-pred;
#               N=16 already exists as theta_sweep_1025guard/th1.5)
# LPT family already measured in results/lpt_1027/. ASCII-only, resumable.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_strategy_cmp_1027.ps1

$ErrorActionPreference = 'Continue'
Set-Location (Join-Path $PSScriptRoot '..')

$common = @(
    'experiments/exp_cooling_distance_scan.py',
    '--distances', '3', '5', '7',
    '--cool-us', '1000', '--coolant-recool-us', '1000',
    '--ancilla-cool-us', '1000', '--cool-residual', '0',
    '--params', 'params/mainz_heating.json',
    '--shots', '100000', '--quiet-clip'
)

$points = @(
    @{ dir = 'anc';        extra = @('--arms', 'anc') },
    @{ dir = 'pred_x1';    extra = @('--arms', 'pred', '--pred-threshold', '1.5', '--pred-max-iters', '1') },
    @{ dir = 'pred_x4';    extra = @('--arms', 'pred', '--pred-threshold', '1.5', '--pred-max-iters', '4') },
    @{ dir = 'pred_x16';   extra = @('--arms', 'pred', '--pred-threshold', '1.5', '--pred-max-iters', '16') },
    @{ dir = 'resident';   extra = @('--arms', 'resident') },
    @{ dir = 'respred_x1'; extra = @('--arms', 'resident-pred', '--pred-threshold', '1.5', '--pred-max-iters', '1') },
    @{ dir = 'respred_x4'; extra = @('--arms', 'resident-pred', '--pred-threshold', '1.5', '--pred-max-iters', '4') }
)
foreach ($p in $points) {
    $outDir = "results/strategy_cmp_1027/$($p.dir)"
    if (Test-Path "$outDir/*.json") {
        Write-Output "skip (done): $outDir"
        continue
    }
    Write-Output "=== run: $outDir ==="
    python @common @($p.extra) --out-dir $outDir
}
Write-Output "STRATEGY CMP DONE"

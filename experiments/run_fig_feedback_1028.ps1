# Figure-feedback runs (sensei 10/28): extend the MPT N_max family to
# {1,2,4,8,16} and push the LIZ-PT theta family further down (0.5, 0.35,
# 0.2) for both isolation variants, to see how close d=7 gets to the
# no-heating lower bound. Unified convention (t = 1 ms, nbar_res = 0,
# mainz, 100k shots). ASCII-only, resumable.
#
# Usage:  powershell -ExecutionPolicy Bypass -File experiments/run_fig_feedback_1028.ps1

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
    @{ out = 'results/strategy_cmp_1027/pred_x2';
       extra = @('--arms', 'pred', '--pred-threshold', '1.5', '--pred-max-iters', '2') },
    @{ out = 'results/strategy_cmp_1027/pred_x8';
       extra = @('--arms', 'pred', '--pred-threshold', '1.5', '--pred-max-iters', '8') },
    @{ out = 'results/strategy_cmp_1027/respred_x2';
       extra = @('--arms', 'resident-pred', '--pred-threshold', '1.5', '--pred-max-iters', '2') },
    @{ out = 'results/strategy_cmp_1027/respred_x8';
       extra = @('--arms', 'resident-pred', '--pred-threshold', '1.5', '--pred-max-iters', '8') },
    @{ out = 'results/lpt_1027/temp_th0.5x4';
       extra = @('--arms', 'resident', '--liz-pt', '0.5', '--liz-pt-iters', '4', '--liz-pt-park', 'temp') },
    @{ out = 'results/lpt_1027/temp_th0.35x4';
       extra = @('--arms', 'resident', '--liz-pt', '0.35', '--liz-pt-iters', '4', '--liz-pt-park', 'temp') },
    @{ out = 'results/lpt_1027/temp_th0.2x4';
       extra = @('--arms', 'resident', '--liz-pt', '0.2', '--liz-pt-iters', '4', '--liz-pt-park', 'temp') },
    @{ out = 'results/lpt_1027/memory_th0.5x4';
       extra = @('--arms', 'resident', '--liz-pt', '0.5', '--liz-pt-iters', '4', '--liz-pt-park', 'memory') },
    @{ out = 'results/lpt_1027/memory_th0.35x4';
       extra = @('--arms', 'resident', '--liz-pt', '0.35', '--liz-pt-iters', '4', '--liz-pt-park', 'memory') },
    @{ out = 'results/lpt_1027/memory_th0.2x4';
       extra = @('--arms', 'resident', '--liz-pt', '0.2', '--liz-pt-iters', '4', '--liz-pt-park', 'memory') }
)
foreach ($p in $points) {
    if (Test-Path "$($p.out)/*.json") {
        Write-Output "skip (done): $($p.out)"
        continue
    }
    Write-Output "=== run: $($p.out) ==="
    python @common @($p.extra) --out-dir $p.out
}
Write-Output "FIG FEEDBACK RUNS DONE"

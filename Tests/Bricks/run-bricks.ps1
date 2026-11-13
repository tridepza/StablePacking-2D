param([uint32]$Seed)

$runtimeDir = Join-Path $PSScriptRoot 'result_smoke_runtime'
$runner = Join-Path $runtimeDir 'run_smoke.py'
$pythonPathFile = Join-Path $runtimeDir 'python_path.txt'
$python = (Get-Content -LiteralPath $pythonPathFile -Raw -ErrorAction Stop).Trim()
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python not found at $python. Update $pythonPathFile with the local Python path."
}

$runArgs = @('-B', '-u', $runner, '--compat', '--highs')
if ($PSBoundParameters.ContainsKey('Seed')) {
    $runArgs += @('--seed', $Seed.ToString())
}

$previousOmpThreads = $env:OMP_NUM_THREADS
$previousOpenBlasThreads = $env:OPENBLAS_NUM_THREADS
try {
    $env:OMP_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    & $python @runArgs
    $runExitCode = $LASTEXITCODE
} finally {
    $env:OMP_NUM_THREADS = $previousOmpThreads
    $env:OPENBLAS_NUM_THREADS = $previousOpenBlasThreads
}
exit $runExitCode

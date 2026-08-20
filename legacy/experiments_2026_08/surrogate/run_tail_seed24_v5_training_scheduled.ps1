$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$python = 'D:\anaconda3\envs\LLM-UAV\python.exe'
$taskName = 'UAVRL-TailSeed24-Train'
$manifest = Join-Path $repository 'artifacts\data\codellama_surrogate_tail_seed24_v5_manifest.json'
$logs = Join-Path $repository 'artifacts\logs'

if (-not (Test-Path -LiteralPath $manifest)) {
    throw 'Seed24 manifest is not ready; training was not started.'
}

$env:PYTHONPATH = 'src'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$training = Start-Process `
    -WindowStyle Hidden `
    -FilePath $python `
    -ArgumentList '-u', 'scripts\train_tail_seed24_v5.py', '--device', 'cuda' `
    -WorkingDirectory $repository `
    -RedirectStandardOutput (Join-Path $logs 'tail_seed24_v5_scheduled_train_retry_stdout.log') `
    -RedirectStandardError (Join-Path $logs 'tail_seed24_v5_scheduled_train_retry_stderr.log') `
    -Wait `
    -PassThru

$exitCode = $training.ExitCode
& 'C:\WINDOWS\system32\schtasks.exe' /Delete /TN $taskName /F | Out-Null
exit $exitCode

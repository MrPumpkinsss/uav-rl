$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$python = 'D:\anaconda3\envs\LLM-UAV\python.exe'
$taskName = 'UAVRL-TailSeed24-V5'
$logs = Join-Path $repository 'artifacts\logs'

$env:PYTHONPATH = 'src'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

New-Item -ItemType Directory -Force -Path $logs | Out-Null
Set-Location $repository

$dataArguments = @(
    '-u',
    'scripts\build_tail_seed24_v5.py',
    '--device', 'cuda',
    '--progress-interval', '500'
)
$dataProcess = Start-Process `
    -WindowStyle Hidden `
    -FilePath $python `
    -ArgumentList $dataArguments `
    -WorkingDirectory $repository `
    -RedirectStandardOutput (Join-Path $logs 'tail_seed24_v5_scheduled_data_stdout.log') `
    -RedirectStandardError (Join-Path $logs 'tail_seed24_v5_scheduled_data_stderr.log') `
    -Wait `
    -PassThru

if ($dataProcess.ExitCode -ne 0) {
    exit $dataProcess.ExitCode
}

$trainArguments = @(
    '-u',
    'scripts\train_tail_seed24_v5.py',
    '--device', 'cuda'
)
$trainProcess = Start-Process `
    -WindowStyle Hidden `
    -FilePath $python `
    -ArgumentList $trainArguments `
    -WorkingDirectory $repository `
    -RedirectStandardOutput (Join-Path $logs 'tail_seed24_v5_scheduled_train_stdout.log') `
    -RedirectStandardError (Join-Path $logs 'tail_seed24_v5_scheduled_train_stderr.log') `
    -Wait `
    -PassThru

$exitCode = $trainProcess.ExitCode
& 'C:\WINDOWS\system32\schtasks.exe' /Delete /TN $taskName /F | Out-Null
exit $exitCode

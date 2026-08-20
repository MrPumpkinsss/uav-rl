param(
    [Parameter(Mandatory = $true)]
    [int]$DataProcessId,
    [string]$TaskName = ''
)

$ErrorActionPreference = 'Stop'
$repository = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $repository 'artifacts/data/codellama_surrogate_tail_seed24_v5_manifest.json'
$python = 'D:\anaconda3\envs\LLM-UAV\python.exe'

try {
    Wait-Process -Id $DataProcessId
    if (-not (Test-Path -LiteralPath $manifest)) {
        throw 'Seed24 data process ended without producing its manifest; training was not started.'
    }

    $env:PYTHONPATH = 'src'
    $env:KMP_DUPLICATE_LIB_OK = 'TRUE'
    Push-Location $repository
    try {
        $logs = Join-Path $repository 'artifacts\logs'
        New-Item -ItemType Directory -Force -Path $logs | Out-Null
        $training = Start-Process `
            -WindowStyle Hidden `
            -FilePath $python `
            -ArgumentList '-u', 'scripts\train_tail_seed24_v5.py', '--device', 'cuda' `
            -WorkingDirectory $repository `
            -RedirectStandardOutput (Join-Path $logs 'tail_seed24_v5_followup_train_stdout.log') `
            -RedirectStandardError (Join-Path $logs 'tail_seed24_v5_followup_train_stderr.log') `
            -Wait `
            -PassThru
        if ($training.ExitCode -ne 0) {
            throw ('Seed24 training exited with code {0}.' -f $training.ExitCode)
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($TaskName) {
        & 'C:\WINDOWS\system32\schtasks.exe' /Delete /TN $TaskName /F | Out-Null
    }
}

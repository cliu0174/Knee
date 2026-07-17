param(
    [string]$OutputDir = "data/severson_raw"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Severson/MIT-Stanford raw dataset download helper"
Write-Host "Output directory: $OutputDir"
Write-Host ""
Write-Host "Official MATR page:"
Write-Host "  https://data.matr.io/1/projects/5c48dd2bc625d700019f3204"
Write-Host ""
Write-Host "Kaggle mirror:"
Write-Host "  https://www.kaggle.com/datasets/rickandjoe/mit-battery-degradation-dataset"
Write-Host ""
Write-Host "Expected raw files:"
Write-Host "  2017-05-12_batchdata_updated_struct_errorcorrect.mat  (~3.03 GB)"
Write-Host "  2017-06-30_batchdata_updated_struct_errorcorrect.mat  (~2.01 GB)"
Write-Host "  2018-04-12_batchdata_updated_struct_errorcorrect.mat  (~3.24 GB)"
Write-Host ""

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Host "Kaggle CLI is not installed or not on PATH."
    Write-Host ""
    Write-Host "Install and configure it with:"
    Write-Host "  pip install kaggle"
    Write-Host "  mkdir `$env:USERPROFILE\.kaggle"
    Write-Host "  # put kaggle.json from your Kaggle account into `$env:USERPROFILE\.kaggle\kaggle.json"
    Write-Host ""
    Write-Host "Then rerun:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/download_severson_raw_data.ps1"
    exit 1
}

kaggle datasets download -d rickandjoe/mit-battery-degradation-dataset -p $OutputDir --unzip

Write-Host ""
Write-Host "Downloaded files:"
Get-ChildItem -Path $OutputDir -File | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize

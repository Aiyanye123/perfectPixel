$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "未找到 .venv，请先创建虚拟环境并安装依赖。"
}

& $Python -m PyInstaller --noconfirm --clean (Join-Path $Root "perfect_pixel_web.spec")

$Target = Join-Path $Root "portable"
if (Test-Path $Target) {
    Remove-Item -LiteralPath $Target -Recurse -Force
}
New-Item -ItemType Directory -Path $Target | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "dist\PerfectPixel-Web.exe") -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root "PORTABLE_README.txt") -Destination $Target
Write-Host "便携版已生成：$Target"

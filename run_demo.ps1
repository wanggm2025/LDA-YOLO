$ErrorActionPreference = "Stop"

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

& $Python (Join-Path $PSScriptRoot "make_demo_data.py") --size 128
& $Python (Join-Path $PSScriptRoot "train.py") --data (Join-Path $PSScriptRoot "configs\demo.yaml") --epochs 2 --img-size 128 --batch 4 --base-channels 8 --output (Join-Path $PSScriptRoot "runs\train\demo")
& $Python (Join-Path $PSScriptRoot "evaluate.py") --weights (Join-Path $PSScriptRoot "runs\train\demo\best.pt") --data (Join-Path $PSScriptRoot "configs\demo.yaml") --batch 4
& $Python (Join-Path $PSScriptRoot "predict.py") --weights (Join-Path $PSScriptRoot "runs\train\demo\best.pt") --source (Join-Path $PSScriptRoot "demo_data\images\val\0000.jpg") --output (Join-Path $PSScriptRoot "runs\predict\demo.jpg") --conf 0.05


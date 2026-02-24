$ErrorActionPreference = "Stop"

Write-Host "[1/3] Python compile check..."
python -m compileall src\scripts\py_files src\interface

Write-Host "[2/3] Python smoke tests..."
python -m unittest discover -s tests -p "test_*.py" -v

Write-Host "[3/3] JavaScript syntax checks..."
Get-ChildItem -Path "src\scripts\js_files" -Filter "*.js" | ForEach-Object {
    node --check $_.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "JavaScript syntax check failed: $($_.FullName)"
    }
}

Write-Host "All local checks passed."

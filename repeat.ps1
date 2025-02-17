while ($true) {
    Write-Host "Launching Simulator..."
    Start-Process "target\release\cruise.exe" -Wait
}
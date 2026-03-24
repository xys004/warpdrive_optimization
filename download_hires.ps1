New-Item -ItemType Directory -Force "C:\Users\Nelson\Downloads\warp_optimization\hires_maps\domain2" | Out-Null
gsutil -m cp "gs://warpopt-data/hires_maps/domain2/*" "C:\Users\Nelson\Downloads\warp_optimization\hires_maps\domain2\"
Write-Host "Downloaded:"
Get-ChildItem "C:\Users\Nelson\Downloads\warp_optimization\hires_maps\domain2" | Select-Object Name, Length | Format-Table -AutoSize

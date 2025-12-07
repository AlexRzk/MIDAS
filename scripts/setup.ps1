# MIDAS Pipeline Setup Script (Windows)
# Run this script to initialize the project

Write-Host "===================================" -ForegroundColor Cyan
Write-Host "MIDAS - Market Intelligence Data Acquisition System" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan

# Create data directories
Write-Host "Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\clean" | Out-Null
New-Item -ItemType Directory -Force -Path "data\features" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\collector" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\processor" | Out-Null
New-Item -ItemType Directory -Force -Path "logs\features" | Out-Null

# Copy environment file if not exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env to configure your settings" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the pipeline:" -ForegroundColor Cyan
Write-Host "  docker compose up -d" -ForegroundColor White
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Cyan
Write-Host "  docker compose logs -f collector" -ForegroundColor White
Write-Host "  docker compose logs -f processor" -ForegroundColor White
Write-Host "  docker compose logs -f features" -ForegroundColor White
Write-Host ""
Write-Host "To stop the pipeline:" -ForegroundColor Cyan
Write-Host "  docker compose down" -ForegroundColor White
Write-Host ""

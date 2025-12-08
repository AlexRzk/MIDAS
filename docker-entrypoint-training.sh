#!/bin/bash
# Entrypoint script for MIDAS training container
set -e

echo "==================================================="
echo "MIDAS Training Container"
echo "==================================================="
echo ""

# Check CUDA availability
if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
else
    echo "WARNING: CUDA not available, will use CPU"
fi

echo ""

# Check data
if [ -d "/app/data/features" ] && [ "$(ls -A /app/data/features 2>/dev/null)" ]; then
    echo "Found feature files:"
    ls -lh /app/data/features/*.parquet 2>/dev/null | head -5
    echo ""
else
    echo "WARNING: No feature files found in /app/data/features"
    echo "Mount your data directory: -v /path/to/data:/app/data"
    echo ""
fi

# Execute command
exec "$@"

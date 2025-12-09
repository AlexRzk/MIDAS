#!/bin/bash
# Fix normalization issues and re-run environment check
# Run this on the VAST remote host

set -e  # Exit on error

echo "============================================"
echo " MIDAS Normalization Fix Script"
echo "============================================"
echo ""

# 1. Update feature schema (already done in local repo - need to sync)
echo "📦 Step 1: Sync code changes from local repo"
echo "   ACTION NEEDED: On your local machine, run:"
echo "   git add -A && git commit -m 'Fix microprice normalization' && git push"
echo "   Then on this remote host, run:"
echo "   git pull origin main"
echo ""
read -p "Press Enter after you've pulled the latest code..."

# 2. Backup old scalers
echo ""
echo "💾 Step 2: Backup old scalers"
if [ -d "data/scalers" ]; then
    BACKUP_DIR="data/scalers_backup_$(date +%Y%m%d_%H%M%S)"
    cp -r data/scalers "$BACKUP_DIR"
    echo "   ✓ Backed up to: $BACKUP_DIR"
else
    echo "   ⚠ No existing scalers found"
fi

# 3. Remove old normalized data
echo ""
echo "🗑️  Step 3: Clean old normalized data"
if [ -d "data/features_normalized" ]; then
    rm -rf data/features_normalized
    echo "   ✓ Removed old normalized features"
fi

# 4. Re-run normalization with fixed schema
echo ""
echo "🔧 Step 4: Re-normalize features with corrected schema"
echo "   This will:"
echo "   - Exclude microprice from normalization (keep raw price)"
echo "   - Fit scalers on training data only"
echo "   - Validate distributions"

python3 scripts/normalize_existing_features.py \
    --input-dir data/features \
    --scaler-dir data/scalers \
    --output-dir data/features_normalized \
    --overwrite

if [ $? -eq 0 ]; then
    echo "   ✓ Normalization complete"
else
    echo "   ❌ Normalization failed"
    exit 1
fi

# 5. Verify normalization
echo ""
echo "✅ Step 5: Verify normalized data"
python3 training/gpu_project/run_env_check.py

echo ""
echo "============================================"
echo " Fix Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Check the validation output above"
echo "2. If everything looks good, run training:"
echo "   ./scripts/train_and_backtest.sh --model tft --epochs 50 --gpus 1"
echo ""

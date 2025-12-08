#!/bin/bash
# Complete verification checklist for MIDAS data pipeline
# Run this from the repository root on your Debian machine

set -e

echo "========================================="
echo "MIDAS Data Pipeline Verification"
echo "========================================="
echo ""

# Step 1: Check raw data files
echo "[1] Checking raw data files..."
raw_count=$(find data/raw -name "*.jsonl.zst" 2>/dev/null | wc -l)
if [ "$raw_count" -eq 0 ]; then
    echo "❌ No raw files found in data/raw"
    echo "   The collector may not be running or files may be in a Docker volume."
    echo "   Run: docker volume inspect midas_raw_data"
else
    echo "✅ Found $raw_count raw files"
    latest_raw=$(ls -t data/raw/*.jsonl.zst 2>/dev/null | head -n1)
    echo "   Latest: $(basename $latest_raw)"
fi
echo ""

# Step 2: Check clean data files
echo "[2] Checking clean data files..."
clean_count=$(find data/clean -name "*.parquet" 2>/dev/null | wc -l)
if [ "$clean_count" -eq 0 ]; then
    echo "❌ No clean files found in data/clean"
    echo "   The processor may not have run yet."
else
    echo "✅ Found $clean_count clean files"
    latest_clean=$(ls -t data/clean/*.parquet 2>/dev/null | head -n1)
    if [ -n "$latest_clean" ]; then
        echo "   Latest: $(basename $latest_clean)"
        echo "   Columns:"
        python3 scripts/inspect_parquet.py "$latest_clean" --show-columns 2>/dev/null | head -n 20
    fi
fi
echo ""

# Step 3: Check feature files
echo "[3] Checking feature data files..."
feature_count=$(find data/features -name "*.parquet" 2>/dev/null | wc -l)
if [ "$feature_count" -eq 0 ]; then
    echo "❌ No feature files found in data/features"
    echo "   The features service may not have run yet."
else
    echo "✅ Found $feature_count feature files"
    latest_feature=$(ls -t data/features/*.parquet 2>/dev/null | head -n1)
    if [ -n "$latest_feature" ]; then
        echo "   Latest: $(basename $latest_feature)"
        echo "   Row count and column sample:"
        python3 scripts/inspect_parquet.py "$latest_feature" 2>/dev/null | head -n 30
    fi
fi
echo ""

# Step 4: Check for advanced features
echo "[4] Checking for advanced features in latest feature file..."
if [ -n "$latest_feature" ]; then
    echo "   Looking for: kyle_lambda, vpin, ladder_slope, queue_imb, vol_of_vol"
    python3 scripts/inspect_parquet.py "$latest_feature" --show-columns 2>/dev/null | grep -E 'kyle_lambda|vpin|ladder_slope|queue_imb|vol_of_vol' || echo "   ⚠️  Advanced features NOT found"
fi
echo ""

# Step 5: Check processor .processed_files
echo "[5] Checking processor tracking (.processed_files)..."
if [ -f data/clean/.processed_files ]; then
    proc_count=$(wc -l < data/clean/.processed_files)
    echo "✅ Processor has tracked $proc_count files"
    echo "   Last 5 processed:"
    tail -n 5 data/clean/.processed_files
else
    echo "⚠️  No .processed_files found (processor may be fresh or tracking disabled)"
fi
echo ""

# Step 6: Check features .processed_files
echo "[6] Checking features tracking (.processed_files)..."
if [ -f data/features/.processed_files ]; then
    feat_proc_count=$(wc -l < data/features/.processed_files)
    echo "✅ Features service has tracked $feat_proc_count files"
    echo "   Last 5 processed:"
    tail -n 5 data/features/.processed_files
else
    echo "⚠️  No .processed_files found (features may be fresh or tracking disabled)"
fi
echo ""

# Step 7: Check Docker containers status
echo "[7] Checking Docker containers..."
docker compose ps
echo ""

# Step 8: Check recent logs for processing activity
echo "[8] Checking recent processor logs for file_processed..."
docker compose logs --tail=50 processor 2>/dev/null | grep -E 'file_processed|processing_file' || echo "   No recent processing logs found"
echo ""

echo "[9] Checking recent features logs for wrote_feature_file..."
docker compose logs --tail=50 features 2>/dev/null | grep -E 'wrote_feature_file|processing_file' || echo "   No recent feature generation logs found"
echo ""

# Step 10: Summary and recommendations
echo "========================================="
echo "Summary & Next Steps"
echo "========================================="

if [ "$clean_count" -eq 0 ]; then
    echo "❌ Processor has not generated clean files yet"
    echo "   Action: Check processor logs with: docker compose logs processor"
elif [ "$feature_count" -eq 0 ]; then
    echo "❌ Features service has not generated feature files yet"
    echo "   Action: Check features logs with: docker compose logs features"
else
    echo "✅ Pipeline is running and has generated files"
    echo ""
    echo "To verify new advanced features are present:"
    echo "  python3 scripts/inspect_parquet.py $latest_feature --show-columns | grep -E 'kyle|vpin|ladder|queue|vol_of_vol'"
    echo ""
    echo "To check if old files were reprocessed:"
    echo "  Compare timestamps: ls -lt data/features/*.parquet | head -n 10"
    echo "  Check .processed_files: tail -n 20 data/features/.processed_files"
    echo ""
    echo "To generate klines from trade data:"
    echo "  python3 scripts/generate_klines_from_trades.py --interval 1m --source raw"
fi

echo ""
echo "Done!"

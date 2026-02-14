#!/bin/bash
# Run all strategies one by one for a given timeframe
# Usage: bash opt_run_all.sh 2h  (or 4h)
TF=${1:-2h}
STRATS="T1 M1 RSI MACD SMA MOM DIP DON BKD STR MP"

echo "=== Running $TF optimization ==="
echo "Strategies: $STRATS"

for S in $STRATS; do
    echo ""
    python -u opt_single.py $TF $S
done

echo ""
echo "=== All IS sweeps done for $TF ==="
echo "Running OOS validation..."
python -u opt_combine.py $TF
echo "=== $TF COMPLETE ==="

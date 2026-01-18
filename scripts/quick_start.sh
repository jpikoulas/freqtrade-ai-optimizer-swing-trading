#!/bin/bash
# Quick start script - Run a single backtest to verify setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "FreqTrade Quick Start Test"
echo "=========================================="

# Build images
echo "Building Docker images..."
sudo docker compose build freqtrade

# Download minimal data (7 days)
echo "Downloading 7 days of data for quick test..."
sudo docker compose run --rm freqtrade download-data \
    --config /freqtrade/config/freqtrade_config.json \
    --pairs BTC/USDT ETH/USDT \
    --timeframe 1m 5m 15m \
    --days 7

# Copy initial strategy
cp strategies/FreqAIStrategy.py user_data/strategies/

# Run a quick backtest
echo "Running quick backtest..."
sudo docker compose run --rm freqtrade backtesting \
    --config /freqtrade/config/freqtrade_config.json \
    --strategy FreqAIStrategy \
    --timerange 20251201-

echo ""
echo "=========================================="
echo "Quick test completed!"
echo "=========================================="
echo ""
echo "If the backtest ran successfully, you can run the full optimizer with:"
echo "  ./run_optimizer.sh"

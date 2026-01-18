#!/bin/bash
# FreqTrade AI Strategy Optimizer - Main run script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FreqTrade AI Strategy Optimizer${NC}"
echo -e "${GREEN}========================================${NC}"

# Check for .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit .env and add your ANTHROPIC_API_KEY${NC}"
    exit 1
fi

# Source environment variables
source .env

# Check for required API key
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your_anthropic_api_key_here" ]; then
    echo -e "${RED}Error: ANTHROPIC_API_KEY not set in .env file${NC}"
    exit 1
fi

# Default values
MAX_ITERATIONS=${MAX_ITERATIONS:-25}
TARGET_PROFIT=${TARGET_PROFIT:-5.0}
BACKTEST_DAYS=${BACKTEST_DAYS:-60}

echo ""
echo "Configuration:"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Target Profit: $TARGET_PROFIT%"
echo "  Backtest Days: $BACKTEST_DAYS"
echo ""

# Build Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose build

# Download data first
echo -e "${YELLOW}Downloading historical data (this may take a few minutes)...${NC}"
docker-compose --profile download run --rm download-data

# Run the optimizer
echo -e "${GREEN}Starting optimization loop...${NC}"
echo ""
docker-compose run --rm optimizer

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Optimization completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results:"
    echo "  - Strategies saved in: user_data/strategies/"
    echo "  - Report: user_data/optimization_report.json"
    echo "  - Logs: user_data/optimizer.log"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Optimization did not reach target profit${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the best strategy in: user_data/strategies/"
fi

#!/bin/bash
#
# FreqTrade AI Strategy Optimizer - Docker Runner
# Runs the entire optimization cycle in containers
#
# Usage:
#   ./run_optimizer.sh                    # Run with defaults (10 iterations)
#   ./run_optimizer.sh --iterations 30    # Run 30 iterations
#   ./run_optimizer.sh --target 10        # Target 10% profit
#   ./run_optimizer.sh --background       # Run in background (detached)
#   ./run_optimizer.sh --forever          # Run indefinitely until target reached
#
# Requirements:
#   - Docker installed and running
#   - DEEPSEEK_API_KEY in .env file
#

set -e

# Default values
MAX_ITERATIONS=10
TARGET_PROFIT=5.0
BACKTEST_DAYS=90
HYPEROPT_EPOCHS=100
DEEPSEEK_MODEL="deepseek-chat"
RUN_BACKGROUND=false
RUN_FOREVER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_PROFIT="$2"
            shift 2
            ;;
        -d|--days)
            BACKTEST_DAYS="$2"
            shift 2
            ;;
        -e|--epochs)
            HYPEROPT_EPOCHS="$2"
            shift 2
            ;;
        -m|--model)
            DEEPSEEK_MODEL="$2"
            shift 2
            ;;
        -b|--background)
            RUN_BACKGROUND=true
            shift
            ;;
        -f|--forever)
            RUN_FOREVER=true
            # Keep MAX_ITERATIONS at user-specified value (or default)
            # The shell script will loop until target is reached
            shift
            ;;
        --status)
            # Show status of running optimizer
            CONTAINER_ID=$(sudo docker ps -q --filter "name=freqtrade-swing-optimizer")
            if [ -n "$CONTAINER_ID" ]; then
                echo "Optimizer is running (container: $CONTAINER_ID)"
                echo ""
                echo "Recent logs:"
                sudo docker logs --tail 50 "$CONTAINER_ID"
            else
                echo "No optimizer is currently running"
            fi
            exit 0
            ;;
        --logs)
            # Follow logs of running optimizer
            CONTAINER_ID=$(sudo docker ps -q --filter "name=freqtrade-swing-optimizer")
            if [ -n "$CONTAINER_ID" ]; then
                sudo docker logs -f "$CONTAINER_ID"
            else
                echo "No optimizer is currently running"
                echo "Check log file: user_data/optimizer.log"
            fi
            exit 0
            ;;
        --stop)
            # Stop running optimizer
            CONTAINER_ID=$(sudo docker ps -q --filter "name=freqtrade-swing-optimizer")
            if [ -n "$CONTAINER_ID" ]; then
                echo "Stopping optimizer..."
                sudo docker stop "$CONTAINER_ID"
                echo "Optimizer stopped"
            else
                echo "No optimizer is currently running"
            fi
            exit 0
            ;;
        -h|--help)
            echo "FreqTrade AI Strategy Optimizer"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --iterations N    Maximum optimization iterations (default: 10)"
            echo "  -t, --target N        Target profit percentage (default: 5.0)"
            echo "  -d, --days N          Backtest period in days (default: 90)"
            echo "  -e, --epochs N        Hyperopt epochs per iteration (default: 100)"
            echo "  -m, --model NAME      DeepSeek model: deepseek-chat or deepseek-reasoner (default: deepseek-chat)"
            echo "  -b, --background      Run in background (detached mode)"
            echo "  -f, --forever         Run indefinitely until target profit is reached"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Management commands:"
            echo "  --status              Show status of running optimizer"
            echo "  --logs                Follow logs of running optimizer"
            echo "  --stop                Stop running optimizer"
            echo ""
            echo "Examples:"
            echo "  $0 --iterations 30 --target 10"
            echo "  $0 -i 20 -m deepseek-reasoner"
            echo "  $0 --forever --background    # Run until 5% profit, in background"
            echo "  $0 --target 15 --forever -b  # Run until 15% profit, in background"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please create .env with your DEEPSEEK_API_KEY:"
    echo "  echo 'DEEPSEEK_API_KEY=your-key-here' > .env"
    exit 1
fi

# Load API key from .env
source .env
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "ERROR: DEEPSEEK_API_KEY not set in .env file!"
    exit 1
fi

# Check Docker is running
if ! sudo docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running!"
    exit 1
fi

# Check if optimizer is already running
EXISTING_CONTAINER=$(sudo docker ps -q --filter "name=freqtrade-swing-optimizer")
if [ -n "$EXISTING_CONTAINER" ]; then
    echo "WARNING: Optimizer is already running (container: $EXISTING_CONTAINER)"
    echo "Use --stop to stop it first, or --logs to view its output"
    exit 1
fi

# Build freqtrade image using docker compose if needed
echo "Building freqtrade image..."
sudo docker compose build freqtrade

# Build the optimizer image if needed
OPTIMIZER_IMAGE="freqtrade-swing-optimizer-runner:latest"

echo "============================================================"
echo "FreqTrade SWING TRADING Optimizer"
echo "============================================================"
echo "Settings:"
echo "  Max Iterations:  $MAX_ITERATIONS"
echo "  Target Profit:   $TARGET_PROFIT%"
echo "  Backtest Days:   $BACKTEST_DAYS"
echo "  Hyperopt Epochs: $HYPEROPT_EPOCHS"
echo "  AI Model:        $DEEPSEEK_MODEL"
if [ "$RUN_FOREVER" = true ]; then
    echo "  Mode:            Run forever until target reached"
fi
if [ "$RUN_BACKGROUND" = true ]; then
    echo "  Background:      Yes (logs in user_data/optimizer.log)"
fi
echo "============================================================"
echo ""

# Check if optimizer image exists or if run_local.py is newer
BUILD_IMAGE=false
if ! sudo docker image inspect "$OPTIMIZER_IMAGE" > /dev/null 2>&1; then
    BUILD_IMAGE=true
elif [ "run_local.py" -nt ".optimizer_image_built" ] 2>/dev/null; then
    BUILD_IMAGE=true
fi

if [ "$BUILD_IMAGE" = true ]; then
    echo "Building optimizer Docker image..."

    # Create Dockerfile for optimizer
    cat > Dockerfile.optimizer << 'DOCKERFILE'
FROM python:3.11-slim

# Install Docker CLI and docker-compose for running freqtrade containers
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    sudo \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Allow running docker commands with sudo without password
RUN echo "ALL ALL=(ALL) NOPASSWD: /usr/bin/docker" >> /etc/sudoers

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    openai \
    python-dotenv

# Copy optimizer script and docker-compose.yml
COPY run_local.py /app/
COPY strategies /app/strategies/
COPY docker-compose.yml /app/

CMD ["python", "-u", "run_local.py"]
DOCKERFILE

    sudo docker build -f Dockerfile.optimizer -t "$OPTIMIZER_IMAGE" .
    rm -f Dockerfile.optimizer
    touch .optimizer_image_built
    echo "Optimizer image built successfully"
fi

# Ensure directories exist
mkdir -p user_data/strategies
mkdir -p user_data/backtest_results
mkdir -p user_data/hyperopt_results
mkdir -p user_data/data
mkdir -p user_data/strategy_backups

# Copy initial strategy if not exists
if [ ! -f "user_data/strategies/FreqAIStrategy.py" ]; then
    cp strategies/FreqAIStrategy.py user_data/strategies/
fi

# Build docker run command
DOCKER_CMD="sudo docker run --rm \
    --name freqtrade-swing-optimizer \
    -v $SCRIPT_DIR/user_data:/app/user_data \
    -v $SCRIPT_DIR/config:/app/config \
    -v $SCRIPT_DIR/strategies:/app/strategies \
    -v $SCRIPT_DIR/docker-compose.yml:/app/docker-compose.yml \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
    -e MAX_ITERATIONS=$MAX_ITERATIONS \
    -e TARGET_PROFIT=$TARGET_PROFIT \
    -e BACKTEST_DAYS=$BACKTEST_DAYS \
    -e HYPEROPT_EPOCHS=$HYPEROPT_EPOCHS \
    -e DEEPSEEK_MODEL=$DEEPSEEK_MODEL \
    -e PROJECT_DIR=/app \
    -e HOST_PROJECT_DIR=$SCRIPT_DIR \
    --network host \
    $OPTIMIZER_IMAGE"

# Function to check if target profit was achieved
check_target_achieved() {
    if [ -f "$SCRIPT_DIR/user_data/optimization_report.json" ]; then
        # Extract best_profit from the report
        BEST_PROFIT=$(python3 -c "import json; r=json.load(open('$SCRIPT_DIR/user_data/optimization_report.json')); print(r.get('summary',{}).get('best_profit', -999))" 2>/dev/null || echo "-999")
        # Compare with target (using python for float comparison)
        TARGET_MET=$(python3 -c "print('yes' if float('$BEST_PROFIT') >= float('$TARGET_PROFIT') else 'no')" 2>/dev/null || echo "no")
        if [ "$TARGET_MET" = "yes" ]; then
            return 0  # Target achieved
        fi
    fi
    return 1  # Target not achieved
}

# Function to run one optimization cycle
run_optimization_cycle() {
    eval "$DOCKER_CMD"
    return $?
}

if [ "$RUN_BACKGROUND" = true ]; then
    echo "Starting optimization in background..."
    echo "Logs will be written to: user_data/optimizer.log"
    echo ""

    if [ "$RUN_FOREVER" = true ]; then
        # Run forever loop in background
        nohup bash -c "
            cd '$SCRIPT_DIR'
            CYCLE=1
            while true; do
                echo ''
                echo '============================================================'
                echo \"FOREVER MODE - Cycle \$CYCLE started at \$(date)\"
                echo '============================================================'

                # Run the optimization
                $DOCKER_CMD
                EXIT_CODE=\$?

                # Check if target was achieved
                if [ -f 'user_data/optimization_report.json' ]; then
                    BEST_PROFIT=\$(python3 -c \"import json; r=json.load(open('user_data/optimization_report.json')); print(r.get('summary',{}).get('best_profit', -999))\" 2>/dev/null || echo '-999')
                    TARGET_MET=\$(python3 -c \"print('yes' if float('\$BEST_PROFIT') >= float('$TARGET_PROFIT') else 'no')\" 2>/dev/null || echo 'no')

                    echo \"Cycle \$CYCLE complete. Best profit: \$BEST_PROFIT% (target: $TARGET_PROFIT%)\"

                    if [ \"\$TARGET_MET\" = 'yes' ]; then
                        echo ''
                        echo '============================================================'
                        echo 'TARGET PROFIT ACHIEVED! Stopping forever loop.'
                        echo '============================================================'
                        exit 0
                    fi
                fi

                echo \"Target not reached. Starting cycle \$((CYCLE + 1)) in 10 seconds...\"
                sleep 10
                CYCLE=\$((CYCLE + 1))
            done
        " > "$SCRIPT_DIR/user_data/optimizer.log" 2>&1 &
    else
        # Run single cycle in background
        nohup bash -c "$DOCKER_CMD" > "$SCRIPT_DIR/user_data/optimizer.log" 2>&1 &
    fi

    # Wait a moment and check if it started
    sleep 2
    CONTAINER_ID=$(sudo docker ps -q --filter "name=freqtrade-swing-optimizer")
    if [ -n "$CONTAINER_ID" ]; then
        echo "Optimizer started successfully (container: $CONTAINER_ID)"
        if [ "$RUN_FOREVER" = true ]; then
            echo "Running in FOREVER mode - will restart until $TARGET_PROFIT% profit is achieved"
        fi
        echo ""
        echo "Management commands:"
        echo "  ./run_optimizer.sh --status   # Check status"
        echo "  ./run_optimizer.sh --logs     # View live logs"
        echo "  ./run_optimizer.sh --stop     # Stop optimizer"
        echo "  tail -f user_data/optimizer.log  # Follow log file"
    else
        echo "ERROR: Failed to start optimizer"
        echo "Check user_data/optimizer.log for details"
        exit 1
    fi
else
    echo "Starting optimization..."
    echo ""

    if [ "$RUN_FOREVER" = true ]; then
        # Run forever loop interactively
        CYCLE=1
        while true; do
            echo ""
            echo "============================================================"
            echo "FOREVER MODE - Cycle $CYCLE started at $(date)"
            echo "============================================================"

            # Run the optimization
            run_optimization_cycle

            # Check if target was achieved
            if check_target_achieved; then
                echo ""
                echo "============================================================"
                echo "TARGET PROFIT ACHIEVED! ($TARGET_PROFIT%)"
                echo "Results saved in: user_data/strategies/"
                echo "Report: user_data/optimization_report.json"
                echo "============================================================"
                exit 0
            fi

            # Get current best profit for display
            BEST_PROFIT=$(python3 -c "import json; r=json.load(open('$SCRIPT_DIR/user_data/optimization_report.json')); print(r.get('summary',{}).get('best_profit', -999))" 2>/dev/null || echo "unknown")
            echo ""
            echo "Cycle $CYCLE complete. Best profit: $BEST_PROFIT% (target: $TARGET_PROFIT%)"
            echo "Target not reached. Starting cycle $((CYCLE + 1)) in 10 seconds..."
            echo "(Press Ctrl+C to stop)"
            sleep 10
            CYCLE=$((CYCLE + 1))
        done
    else
        # Run single cycle interactively
        run_optimization_cycle

        echo ""
        echo "============================================================"
        echo "Optimization complete!"
        echo "Results saved in: user_data/strategies/"
        echo "Report: user_data/optimization_report.json"
        echo "============================================================"
    fi
fi
